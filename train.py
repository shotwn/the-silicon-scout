from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from datasets import load_dataset
import torch
from argparse import ArgumentParser
from numeric_fusion_adapter import NumericFusionAdapter, NumericFeatureCollator
from typing import Any, Optional, Union
import os

# Check environment
arg_parser = ArgumentParser()
arg_parser.add_argument("--override_checkpoints", action="store_true", default=False, required=False, help="Whether to continue training from existing checkpoints or override them.")
arg_parser.add_argument("--output_dir", type=str, default="lora_lhco", required=False, help="Directory to save LoRA checkpoints and numeric fusion adapter weights.")
args = arg_parser.parse_args()

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Will Override Existing Checkpoints" if args.override_checkpoints else "Will Resume from Existing Checkpoints")

# Example: using a 7B model
model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # example name

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

max_memory = {
    0: "7.6GB",
    "cpu": "30GB"
}

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory=max_memory,
)

model = prepare_model_for_kbit_training(model)

# Add Numeric Fusion Adapter
numeric_dim = 16 # See data processing for chosen numeric features
hidden_size = model.config.hidden_size
model.numeric_fusion_adapter = NumericFusionAdapter(hidden_size, numeric_dim, dtype=torch.float32, device=model.device).to(model.device) # !Switched to float32 for testing

# Make sure numeric fusion adapter parameters are trainable
print("\nNumeric Fusion Adapter Parameters:")
for name, param in model.numeric_fusion_adapter.named_parameters():
    param.requires_grad_(True)
    print(f"{name}: requires_grad={param.requires_grad}")
print("")

lora_config = LoraConfig(
    r=32,
    lora_alpha=32*4,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.15,
    bias="none"
)

model = get_peft_model(model, lora_config)
# Heck if I know, it was needed to make max_grad_norm work
model.enable_input_require_grads()

# Important: Set tokenizer.pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
ds = load_dataset("json", data_files={"train":"output/train_one_to_one.jsonl", "validation":"output/val_one_to_one.jsonl"})
#ds = load_dataset("json", data_files={"train":"output/train_original_ratio.jsonl", "validation":"output/val_original_ratio.jsonl"})
def format_example(example):
    jets = example["jets"]
    s = "[INST] Classify this event as 'signal' or 'background'.\n"
    s += "jets:\n"
    for i, j in enumerate(jets):
        s += f"  jet{i+1}: P_T={j['P_T']:.10f} eta={j['eta']:.10f} phi={j['phi']:.10f} E={j['E']:.10f} m={j['m']:.10f} n_particles={j['n_particles']} P_T_lead={j['P_T_lead']:.10f}\n"
        for dR_jet, dR_value in j["dR"].items():
            dR_value = dR_value if dR_value is not None else 0.0
            s += f"    dR_{dR_jet}={dR_value:.2f}\n"
    s += f"n_particles: {example['n_particles']} M_jj= {example['M_jj']}[/INST]\n"

    # HF Trainer expects 'labels' key for supervised fine-tuning
    return {
        "input_text": s,
        "labels": example["type"], 
        # !Bugfix: jets, n_particles, M_jj also needed for numeric features extraction
        "jets": example["jets"],
        "n_particles": example["n_particles"], 
        "M_jj": example["M_jj"]
    }

ds = ds.map(format_example, load_from_cache_file=not args.override_checkpoints)

def tokenize_example_refined(example, max_length=512):
    # 1. Construct the full sequence: Prompt + Answer
    # We add a space to ensure tokenization of the answer doesn't merge with the prompt's last token
    prompt = example["input_text"]
    answer = example["labels"] # 'signal' or 'background'
    full_text = prompt + answer

    # 2. Tokenize the Prompt (Input Text) to find its length
    # Use 'add_special_tokens=False' to avoid counting special tokens in the prompt length
    # if the prompt already contains them (like [/INST]).
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False
    )
    prompt_len = len(prompt_tokens["input_ids"])

    # 3. Tokenize the Full Sequence (Prompt + Answer)
    # This generates the input_ids, attention_mask, etc., and handles padding/truncation.
    full_encoding = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    # 4. Construct the Labels Array
    labels = full_encoding["input_ids"].copy()

    # Mask out the prompt tokens with -100
    # The model should not learn from the prompt itself.
    labels[:prompt_len] = [-100] * prompt_len

    # Mask out the padding tokens with -100
    # The padding tokens are at the end, indicated by the attention mask being 0.
    # Note: If the tokenizer pads with EOS, this step is crucial.
    for i in range(max_length):
        if full_encoding["attention_mask"][i] == 0:
            labels[i] = -100
    
    # Assign the final labels array
    full_encoding["labels"] = labels

    # ===== Build numeric vector =====
    # Use fixed order to extract numeric features
    # 0 -> jet1_P_T, 1 -> jet1_eta, 2 -> jet1_phi, 3 -> jet1_E, 4 -> jet1_m, 5 -> jet1_n_particles, 6 -> jet1_P_T_lead
    # 7 -> jet2_P_T, 8 -> jet2_eta, 9 -> jet2_phi, 10 -> jet2_E, 11 -> jet2_m, 12 -> jet2_n_particles, 13 -> jet2_P_T_lead
    # 14 -> n_particles, 15 -> M_jj
    # No 3rd jet info for now
    # No dR info for now
    # So total numeric_dim = 16
    # You can pick as many features as you want; here is an example:
    numeric_vector = [ 0.0 ] * 16  # Initialize with zeros
    jets = example["jets"]
    # Jet 1
    if len(jets) > 0:
        numeric_vector[0] = float(jets[0]["P_T"]) / 1000.0  # Normalize P_T
        numeric_vector[1] = float(jets[0]["eta"]) # eta can be negative
        numeric_vector[2] = float(jets[0]["phi"]) # phi can be negative
        numeric_vector[3] = float(jets[0]["E"]) / 1000.0  # Normalize E
        numeric_vector[4] = float(jets[0]["m"]) / 100.0  # Normalize mass
        numeric_vector[5] = float(jets[0]["n_particles"]) / 100.0  # Normalize n_particles
        numeric_vector[6] = float(jets[0]["P_T_lead"]) / 1000.0  # Normalize P_T_lead

    # Jet 2
    if len(jets) > 1:
        numeric_vector[7] = float(jets[1]["P_T"]) / 1000.0  # Normalize P_T
        numeric_vector[8] = float(jets[1]["eta"]) # eta can be negative
        numeric_vector[9] = float(jets[1]["phi"]) # phi can be negative
        numeric_vector[10] = float(jets[1]["E"]) / 1000.0  # Normalize E
        numeric_vector[11] = float(jets[1]["m"]) / 100.0  # Normalize mass
        numeric_vector[12] = float(jets[1]["n_particles"]) / 100.0  # Normalize n_particles
        numeric_vector[13] = float(jets[1]["P_T_lead"]) / 1000.0  # Normalize P_T_lead

    # Global features
    numeric_vector[14] = float(example["n_particles"]) / 200.0  # Normalize total n_particles
    numeric_vector[15] = float(example["M_jj"]) / 1000.0  # Normalize M_jj

    full_encoding["numeric_features"] = numeric_vector
    full_encoding["my_test_field"] = 12345  # For debugging

    return full_encoding

# Tokenize the dataset
# load_from_cache_file is set based on whether we are overriding checkpoints or not
# If we are overriding, we want to reprocess the data
tokenized_ds = ds.map(
    tokenize_example_refined, 
    batched=False, 
    load_from_cache_file=not args.override_checkpoints, 
    remove_columns=ds["train"].column_names,
)

# Collator is not receiving numeric features if we don't set the format here
# This might be unnecesary now, culprit was "remove_unused_columns" in Trainer args
tokenized_ds.set_format(
    type="torch", 
    columns=["input_ids", "attention_mask", "labels", "numeric_features"],
    output_all_columns=False
)

# Setup training
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16, # simulate larger batch size
    learning_rate=3e-5, # Decrased from 1e-4 to 3e-5
    num_train_epochs=5,
    save_strategy="steps",
    save_steps=100,
    eval_steps=100,
    logging_steps=30,
    gradient_checkpointing=True, # Transformer was warning me to set this.
    gradient_checkpointing_kwargs={'use_reentrant': False},  # Transformer was warning me to set this. Appearently in future versions it will default to False.
    max_grad_norm=1.0, # Added gradient clipping, after custom weighted loss we were having huge grad_norms (>500) in initial checkpoints
    #! Took forever to find. Important to keep numeric_features, make sure to clean other unused columns
    remove_unused_columns=False, 
)

# Use 8-bit Adam optimizer
from bitsandbytes.optim import AdamW8bit
lora_params = [
    {"params": [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad],
     "lr": training_args.learning_rate}
]

# ! Consider weight decay if overfitting occurs
numeric_params = [
    {"params": model.numeric_fusion_adapter.parameters(),
     "lr": training_args.learning_rate * 100}  # e.g. 3e-3 if base LR is 3e-5
]

optimizer = AdamW8bit(lora_params + numeric_params)

# Debug: Print optimizer parameter groups
print("\nOptimizer parameter groups:")
for i, group in enumerate(optimizer.param_groups):
    print(f"Group {i}: LR={group['lr']}, Parameter Tensors (Layers)={len(group['params'])}")

# Debug: Print model parameter counts
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)\n")



# Custom Trainer to integrate numeric adapter
class NumericTrainer(Trainer):
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        # Extract numeric features
        numeric_features = inputs.pop("numeric_features", None)
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")
        labels = inputs.pop("labels")

        # Use public API to get embeddings
        embedding_layer = model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)
        
        # Prepend numeric embeddings if available
        if numeric_features is not None:
            # device = next(model.parameters()).device
            # numeric_features = torch.tensor(numeric_features, dtype=torch.float32, device=device)
            # This is now done in the NumericFeatureCollator.
            # This ensures numeric_features is already a tensor of correct dtype and device.
            # Plus improves performance by avoiding device transfers in the training loop.
            numeric_embeds = model.numeric_fusion_adapter(numeric_features)

            
            # Extend and Fuse the Inputs/Masks
            inputs_embeds = torch.cat([numeric_embeds, inputs_embeds], dim=1)
            attention_mask = torch.cat(
                [torch.ones((attention_mask.shape[0], 1), device=attention_mask.device), attention_mask],
                dim=1
            )
            
            # Extend and Mask the Labels
            # Create a -100 tensor (mask) of shape (Batch_size, 1)
            numeric_labels_mask = torch.full(
                (labels.shape[0], 1), # (B, 1)
                -100, 
                dtype=labels.dtype, 
                device=labels.device
            )
            # Prepend the mask to the original labels
            labels = torch.cat([numeric_labels_mask, labels], dim=1)

        # Sanity check
        # All sequence lengths should match now
        assert inputs_embeds.shape[1] == attention_mask.shape[1] == labels.shape[1], \
            f"Shape mismatch: embeds {inputs_embeds.shape}, mask {attention_mask.shape}, labels {labels.shape}"

        # Debug: Print adapter weight stats every 100 steps
        if self.state.global_step % 100 == 0:
            adapter = model.numeric_fusion_adapter
            for name, param in adapter.named_parameters():
                if "weight" in name:
                    print(f"[step {self.state.global_step}] {name} mean/std:",
                        param.data.mean().item(), param.data.std().item())
                    if param.grad is not None:
                        print("grad mean/std:",
                            param.grad.mean().item(), param.grad.std().item())
                    break

        # Forward pass
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **inputs  # any extra keys are forwarded safely
        )

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    # Override _save to manually save the custom adapter weights
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # Call the base class save method. This saves the LoRA weights and Trainer state.
        super()._save(output_dir, state_dict)
        
        # Manually save the state dict of the custom adapter
        # This check ensures we only save on the primary process
        if self.args.should_save:
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            
            # Retrieve the adapter module attached to the model
            # Note: self.model is the PeftModel which holds the base model with our adapter attached
            adapter_module = self.model.numeric_fusion_adapter
            
            # Construct the path to save the custom module weights
            adapter_path = os.path.join(output_dir, "numeric_fusion_adapter.bin")
            
            # Save the state dictionary using torch
            torch.save(adapter_module.state_dict(), adapter_path)
            print(f"Custom numeric adapter saved to {adapter_path}")

    def _load_from_checkpoint(self, resume_path):
        # Load standard LoRA weights and Trainer state
        # Call the base class method to handle everything else (LoRA, optimizer, etc.)
        super()._load_from_checkpoint(resume_path)

        # Manually load the state dict of the custom adapter
        adapter_path = os.path.join(resume_path, "numeric_fusion_adapter.bin")

        if os.path.exists(adapter_path):
            print(f"Loading custom numeric adapter from {adapter_path}")
            # Ensure the state_dict is loaded to the correct device
            # map_location='cuda' will load it to the GPU if available
            adapter_state_dict = torch.load(adapter_path, map_location=self.args.device)
            
            # The model attribute here is the PeftModel which holds the base model
            # with our custom adapter attached.
            self.model.numeric_fusion_adapter.load_state_dict(adapter_state_dict)
            print("Successfully loaded custom numeric adapter weights.")
        else:
            print(f"Custom numeric adapter weights not found at {adapter_path}. Starting from scratch/random initialization.")

trainer = NumericTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    optimizers=(optimizer, None),  # use custom 8-bit optimizer
    data_collator=NumericFeatureCollator(),
    # compute_loss_func=compute_loss  # our custom loss function
)

trainer.train(
    resume_from_checkpoint=not args.override_checkpoints
)
