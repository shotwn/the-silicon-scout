from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from datasets import load_dataset
import torch
from argparse import ArgumentParser
from numeric_fusion_adapter import NumericFusionAdapter
from typing import Any, Optional, Union

# Check environment
arg_parser = ArgumentParser()
arg_parser.add_argument("--override_checkpoints", type=bool, default=False, required=False, help="Whether to continue training from existing checkpoints or override them.")
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
    bnb_4bit_compute_dtype="float16"
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
model.numeric_fusion_adapter = NumericFusionAdapter(hidden_size, numeric_dim).to(model.device)


lora_config = LoraConfig(
    r=32,
    lora_alpha=32*4,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.15,
    bias="none"
)

model = get_peft_model(model, lora_config)

# Load dataset
ds = load_dataset("json", data_files={"train":"output/train_one_to_one.jsonl", "validation":"output/val_one_to_one.jsonl"})

def format_example(example):
    jets = example["jets"]
    s = "[INST] Classify this event as 'signal' or 'background'.\n"
    s += "jets:\n"
    for i, j in enumerate(jets):
        s += f"  jet{i+1}: P_T={j['P_T']:.10f} eta={j['eta']:.10f} phi={j['phi']:.10f} E={j['E']:.10f} m={j['m']:.10f} n_particles={j['n_particles']} P_T_lead={j['P_T_lead']:.10f}\n"
        for dR_jet, dR_value in j["dR"].items():
            dR_value = dR_value if dR_value is not None else 0.0
            s += f"    dR_{dR_jet}={dR_value:.2f}\n"
    s += f"n_particles: {example['n_particles']} M_jj= {example['M_jj']}[/INST]"

    # HF Trainer expects 'labels'
    return {"input_text": s, "labels": example["type"]}


ds = ds.map(format_example)

tokenizer.pad_token = tokenizer.eos_token

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
        max_length=max_length
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

    return full_encoding

tokenized_ds = ds.map(tokenize_example_refined, batched=False)

# Heck if I know, it was needed to make max_grad_norm work
model.enable_input_require_grads()

# Setup training
training_args = TrainingArguments(
    output_dir="lora_lhco",
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
    max_grad_norm=1.0 # Added gradient clipping, after custom weighted loss we were having huge grad_norms (>500) in initial checkpoints
)

# Use 8-bit Adam optimizer
from bitsandbytes.optim import AdamW8bit
optimizer = AdamW8bit(model.parameters(), lr=training_args.learning_rate)

# 🧩 Custom Trainer to integrate numeric adapter
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

        # ✅ Use public API to get embeddings
        embedding_layer = model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)

        # Prepend numeric embeddings if available
        if numeric_features is not None:
            numeric_embeds = model.numeric_fusion_adapter(numeric_features.to(model.device))
            
            # 1. Extend and Fuse the Inputs/Masks
            inputs_embeds = torch.cat([numeric_embeds, inputs_embeds], dim=1)
            attention_mask = torch.cat(
                [torch.ones((attention_mask.shape[0], 1), device=attention_mask.device), attention_mask],
                dim=1
            )
            
            # 2. 💡 CRITICAL: Extend and Mask the Labels 💡
            # Create a -100 tensor (mask) of shape (Batch_size, 1)
            numeric_labels_mask = torch.full(
                (labels.shape[0], 1), # (B, 1)
                -100, 
                dtype=labels.dtype, 
                device=labels.device
            )
            # Prepend the mask to the original labels
            labels = torch.cat([numeric_labels_mask, labels], dim=1)

        # Forward pass
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **inputs  # any extra keys are forwarded safely
        )

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

trainer = NumericTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    optimizers=(optimizer, None),  # use custom 8-bit optimizer
    # compute_loss_func=compute_loss  # our custom loss function
)

trainer.train(
    resume_from_checkpoint=not args.override_checkpoints
)
