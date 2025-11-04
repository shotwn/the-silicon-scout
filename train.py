from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers import BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset
import torch
from argparse import ArgumentParser
from numeric_fusion_adapter import NumericFusionAdapter, NumericFeatureCollator
from typing import Any, Optional, Union
import os

# Check environment
arg_parser = ArgumentParser()
arg_parser.add_argument("--override_checkpoints", action="store_true", default=False, required=False, help="Whether to continue training from existing checkpoints or override them.")
arg_parser.add_argument("--reset_dataset_cache", action="store_true", default=False, required=False, help="Whether to reset the dataset cache.")
arg_parser.add_argument("--output_dir", type=str, default="lora_lhco", required=False, help="Directory to save LoRA checkpoints and numeric fusion adapter weights.")
arg_parser.add_argument("--debug_loss_function", action="store_true", default=False, required=False, help="Whether to enable debug print (console) for the loss function.")
args = arg_parser.parse_args()

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Will Override Existing Checkpoints" if args.override_checkpoints else "Will Resume from Existing Checkpoints")

# Example: using a 7B model
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

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

# Base model loading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory=max_memory,
)

model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=32,
    lora_alpha=32*4,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.15,
    bias="none"
)

model = get_peft_model(model, lora_config)

# Add Numeric Fusion Adapter
numeric_dim = 16 # See data processing for chosen numeric features
hidden_size = model.config.hidden_size
numeric_fusion_adapter = NumericFusionAdapter(hidden_size, numeric_dim, dtype=torch.float32, device=model.device).to(model.device) # !Switched to float32 for testing
model.numeric_fusion_adapter = numeric_fusion_adapter

# Make sure numeric fusion adapter parameters are trainable
print("\nNumeric Fusion Adapter Parameters:")
for name, param in model.numeric_fusion_adapter.named_parameters():
    # param.requires_grad_(True)
    print(f"{name}: requires_grad={param.requires_grad}")
print("")

# Heck if I know, it was needed to make max_grad_norm work
model.enable_input_require_grads()

# Important: Set tokenizer.pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

"""
Numeric Fusion Adapter Placeholder Token Initialization
"""
JET_FEATURE_TOKEN_STR = "<JET_FEATURES>"
JET_FEATURE_TOKEN_ID = tokenizer.convert_tokens_to_ids(JET_FEATURE_TOKEN_STR)
# If the token is not already in the tokenizer, add it
if JET_FEATURE_TOKEN_ID is None or JET_FEATURE_TOKEN_ID == tokenizer.unk_token_id:
    # Prepare a custom token for numeric features
    # Add the custom token to the tokenizer
    tokenizer.add_special_tokens({'additional_special_tokens': [JET_FEATURE_TOKEN_STR]})

    # Resize the model embeddings to include the new token
    model.resize_token_embeddings(len(tokenizer))

    # Fetch its actual numeric ID
    JET_FEATURE_TOKEN_ID = tokenizer.convert_tokens_to_ids(JET_FEATURE_TOKEN_STR)
    print(f"Added '{JET_FEATURE_TOKEN_STR}' token with ID: {JET_FEATURE_TOKEN_ID}")
else:
    print(f"'{JET_FEATURE_TOKEN_STR}' token exists with ID: {JET_FEATURE_TOKEN_ID}")

"""
Data Loading and Preprocessing
"""
# Load dataset
#ds = load_dataset("json", data_files={"train":"output/train_one_to_one.jsonl", "validation":"output/val_one_to_one.jsonl"})
ds = load_dataset("json", data_files={"train":"output/train_original_ratio.jsonl", "validation":"output/val_original_ratio.jsonl"})

def format_example(example):
    jets = example["jets"]
    s = "[INST] Classify this event as 'signal' or 'background'.\n"
    s += "jets:\n"
    for i, j in enumerate(jets):
        s += f"  jet{i+1}: P_T={j['P_T']:.10f} eta={j['eta']:.10f} phi={j['phi']:.10f} E={j['E']:.10f} m={j['m']:.10f} n_particles={j['n_particles']} P_T_lead={j['P_T_lead']:.10f}\n"
        for dR_jet, dR_value in j["dR"].items():
            dR_value = dR_value if dR_value is not None else 0.0
            s += f"    dR_{dR_jet}={dR_value:.2f}\n"
    s += f"n_particles: {example['n_particles']} M_jj= {example['M_jj']}{JET_FEATURE_TOKEN_STR}[/INST] "

    # HF Trainer expects 'labels' key for supervised fine-tuning
    return {
        "input_text": s,
        "labels": example["type"], 
        # !Bugfix: jets, n_particles, M_jj also needed for numeric features extraction
        "jets": example["jets"],
        "n_particles": example["n_particles"], 
        "M_jj": example["M_jj"]
    }

ds = ds.map(format_example, load_from_cache_file=not args.reset_dataset_cache)

def tokenize_example_refined(example, max_length=512):
    # 1. Construct the full sequence: Prompt + Answer
    # Prompt ends with space before answer
    # There is a space after the answer to avoid tokenization issues
    # We add a space to ensure tokenization of the answer doesn't merge with the prompt's last token
    prompt = example["input_text"]
    answer = example["labels"] # 'signal' or 'background'
    full_text = f"{prompt}{answer} "

    # 2. Tokenize the Full Sequence (Prompt + Answer)
    # This generates the input_ids, attention_mask, etc., and handles padding/truncation.
    full_encoding = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
        add_special_tokens=False
    )

    # 3. Tokenize the Prompt (Input Text) to find its length
    # Use 'add_special_tokens=False' to avoid counting special tokens in the prompt length
    # if the prompt already contains them (like [/INST]).
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False
    )
    prompt_len = len(prompt_tokens["input_ids"])

    # 4. Construct the Labels Array
    labels = full_encoding["input_ids"].copy()

    # Mask out the padding tokens with -100
    # This model is left-padded, so we find where the left padding ends.
    where_left_padding_ends = 0
    for i in range(max_length):
        if full_encoding["attention_mask"][i] == 0:
            labels[i] = -100
            where_left_padding_ends = i
        else:
            break # First non-padded token index

    # Mask out the prompt tokens with -100
    # The model should not learn from the prompt itself.
    # Mask out prompt tokens (the first part of the non-padded region)
    start_idx = where_left_padding_ends
    end_idx = min(start_idx + prompt_len, max_length)
    labels[start_idx:end_idx] = [-100] * (end_idx - start_idx)

    # Assign the final labels array
    full_encoding["labels"] = labels

    # Print labels (masked) for debugging
    """
    labels_without_mask = [lbl if lbl != -100 else tokenizer.pad_token_id for lbl in labels]
    decoded_labels = tokenizer.decode(labels_without_mask, skip_special_tokens=False)
    print("Decoded labels with masking:", decoded_labels)
    """

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
    load_from_cache_file=not args.reset_dataset_cache, 
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
# * Set run_name properly for logging
run_name = "token_placeholder_NFA_001"
training_args = TrainingArguments(
    run_name=run_name,
    output_dir=args.output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16, # simulate larger batch size
    learning_rate=3e-5, # Decrased from 1e-4 to 3e-5
    num_train_epochs=5,
    save_strategy="steps",
    save_steps=100,
    eval_steps=100,
    logging_steps=36,
    logging_dir=f"./logs/{run_name}",
    gradient_checkpointing=True, # Transformer was warning me to set this. ! Changed to False for testing
    gradient_checkpointing_kwargs={'use_reentrant': False},  # Transformer was warning me to set this. Appearently in future versions it will default to False.
    max_grad_norm=1.0, # Added gradient clipping, after custom weighted loss we were having huge grad_norms (>500) in initial checkpoints
    #! Took forever to find. Important to keep numeric_features, make sure to clean other unused columns
    remove_unused_columns=False, 
    report_to="tensorboard",  # Enable TensorBoard logging
    #label_smoothing_factor=0.1,  
    # Helps prevent overconfidence by softening targets.
    # This makes the model less certain on noisy or imbalanced labels 
    # (like "signal" vs "background" in LHCO) → smoother loss curve.

    #weight_decay=0.003,
    # Adds mild L2 regularization to prevent adapter & LoRA weights 
    # from growing too large. Helps generalization & avoids overfitting.
)



"""
Optimizer Setup with bitsandbytes GlobalOptimManager
"""
# Use 8-bit Adam optimizer
# https://huggingface.co/docs/bitsandbytes/main/en/optimizers
# NOTE: We must import GlobalOptimManager to explicitly set 32-bit optimization for the adapter.
from bitsandbytes.optim import AdamW8bit, GlobalOptimManager

# --- NEW OPTIMIZER SETUP ---
# Allowing 32-bit optimization for the numeric fusion adapter

# Initialize GlobalOptimManager and register the adapter's parameters
mng = GlobalOptimManager.get_instance()
# Register the adapter's parameters before initializing the optimizer
mng.register_parameters(model.numeric_fusion_adapter.parameters())

# Get ALL trainable parameters (LoRA and Adapter)
all_trainable_params = [p for p in model.parameters() if p.requires_grad]

# Initialize the SINGLE Optimizer (AdamW8bit) with all parameters
# The LoRA parameters will use the base LR.
base_lr = training_args.learning_rate
optimizer = AdamW8bit(all_trainable_params, lr=base_lr) 

# Override specific parameters for 32-bit optimization and higher LR
numeric_lr = base_lr * 10 # from 3e-5 → 3e-4 (10× faster for adapter)
numeric_config = {
    'optim_bits': 32, 
    'lr': numeric_lr, 
    #'weight_decay': 0.001 # Slightly lower weight decay for adapter
} 

print("\nApplying GlobalOptimManager overrides for Numeric Fusion Adapter:")
# Override all parameters in the adapter module
for name, param in model.numeric_fusion_adapter.named_parameters():
    # ! Important:
    # Force requires_grad to True, this was reverting to False for some reason
    param.requires_grad_(True)

    print(f"Checking parameter: {name}, requires_grad={param.requires_grad}")

    if not param.requires_grad:
        print(f"Warning: Eventhough forced, parameter {name} has requires_grad=False. Skipping override.")
        continue

    # Use the manager to override the config for this specific parameter
    mng.override_config(param, key_value_dict=numeric_config)
    print(f"- Overrode {name}: optim_bits=32, lr={numeric_lr}")

# Re-print parameter groups for verification (the manager handles the split internally)
print(f"\nTotal trainable params: {len(all_trainable_params)}")

# sanity-check optimizer includes adapter params
missing = []
for name, p in model.numeric_fusion_adapter.named_parameters():
    if not any(p is q for g in optimizer.param_groups for q in g["params"]):
        missing.append((name, p))
if missing:
    print(f"Error: {len(missing)} adapter params missing from optimizer!")
    for name, p in missing:
        print(f" - {name}")
    raise RuntimeError("Some adapter params were not included in optimizer. Recreate optimizer after attaching adapter/PEFT.")


"""
Custom Trainer to integrate numeric adapter
"""
class NumericTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Initialize the base Trainer
        super().__init__(*args, **kwargs)

        # Setup for Weighted Loss Function to handle class imbalance
        #! Following is from Gemini with my touch-ups. Likely won't work properly yet.
        # 1. Get Token IDs for the final labels
        # See debug output in compute_loss to verify these IDs
        # I was sure only "\nsignal" and "\nbackground" would be sufficient but if the prompt changes, wider coverage can be needed.
        white_space_ids = self.processing_class.encode(" ", add_special_tokens=False)
        white_space_ids += self.processing_class.encode("\n", add_special_tokens=False)

        #signal_ids = self.processing_class.encode("signal", add_special_tokens=False)
        #signal_with_space_ids = self.processing_class.encode(" signal", add_special_tokens=False)
        signal_with_newline_ids = self.processing_class.encode(" signal", add_special_tokens=False)

        #self.signal_ids = signal_ids + signal_with_space_ids + signal_with_newline_ids - white_space_ids
        self.signal_ids = [id for id in signal_with_newline_ids if id not in white_space_ids]
        
        #background_ids = self.processing_class.encode("background", add_special_tokens=False)
        #background_with_space_ids = self.processing_class.encode(" background", add_special_tokens=False)
        background_with_newline_ids = self.processing_class.encode(" background", add_special_tokens=False)

        self.background_ids = [id for id in background_with_newline_ids if id not in white_space_ids]

        print(f"Signal token IDs: {self.signal_ids}")
        print(f"Background token IDs: {self.background_ids}")


        # 2. Create Vocab-Sized Weight Tensor
        vocab_size = self.model.config.vocab_size
        # Initialize all tokens to weight 1.0
        # Use CPU first for safety, then move to device, ensure float32
        loss_weights = torch.ones(vocab_size, dtype=torch.float32)

        # 3. Apply Class Weights
        weight_signal = 10.0 # High weight for rare class
        weight_background = 1.0 # Normal weight for common class
        
        # Map token IDs to their weights
        for signal_id in self.signal_ids:
            loss_weights[signal_id] = weight_signal

        for background_id in self.background_ids:
            loss_weights[background_id] = weight_background

        # Move weights to the device
        self.loss_weights = loss_weights.to(self.args.device)

        # Focal Loss gamma parameter
        self.gamma = 1.0  # typical default, can tune 1-3

        # Flag for compute loss sanity check over JET_FEATURE_TOKEN_ID
        self.jet_feature_token_id_sanity_checked = False

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
             # Get the embeddings for the entire input (B, T, hidden_size)
            inputs_embeds = model.get_input_embeddings()(input_ids)

            # Find positions of <JET_FEATURES> tokens
            if not self.jet_feature_token_id_sanity_checked:
                jet_token_id = tokenizer.convert_tokens_to_ids(JET_FEATURE_TOKEN_STR)
                if jet_token_id != JET_FEATURE_TOKEN_ID:
                    print(f"⚠️ Warning: Mismatched {JET_FEATURE_TOKEN_STR} token ID! Expected {JET_FEATURE_TOKEN_ID}, got {jet_token_id}")
                
                self.jet_feature_token_id_sanity_checked = True
            else:
                jet_token_id = JET_FEATURE_TOKEN_ID

            jet_feature_mask = (input_ids == jet_token_id)

            # Sanity check: each row should have exactly 1 match
            if not torch.all(jet_feature_mask.sum(dim=1) == 1):
                print(f"⚠️ Warning: each sample should contain exactly one {JET_FEATURE_TOKEN_STR} token")

            # Replace <JET_FEATURES> embeddings with projected numeric features
            # (works even with batch size > 1)
            # Clone so we don't have leaf variable warnings
            inputs_embeds = inputs_embeds.clone()
            # Replace at masked positions
            inputs_embeds[jet_feature_mask] = numeric_embeds

        # Sanity check
        # All sequence lengths should match now
        assert inputs_embeds.shape[1] == attention_mask.shape[1] == labels.shape[1], \
            f"Shape mismatch: embeds {inputs_embeds.shape}, mask {attention_mask.shape}, labels {labels.shape}"
        
        # make fused embeddings track gradients (needed when using gradient_checkpointing / k-bit wrappers)
        inputs_embeds.requires_grad_(True)

        # Forward pass
        """
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **inputs  # any extra keys are forwarded safely
        )
        """

        # Custom forward without labels to avoid default loss computation
        # We will compute our own weighted loss below
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels, # We will override PART of the loss computation below
            **inputs  # any extra keys are forwarded safely
        )

        original_loss = outputs.loss  # The CE loss HuggingFace computes internally

        if args.debug_loss_function:  # toggle this with self.debug = True/False
            print("\n" + "=" * 80)
            print("🧠 DEBUG MODE: Inspecting input_ids, labels, and logits")
            print("=" * 80)

            # Full predicted token IDs for the batch
            all_logits = outputs.logits.argmax(dim=-1)  # shape [B, L]
            decoded_preds = [
                self.processing_class.decode(seq.tolist(), skip_special_tokens=True)
                for seq in all_logits
            ]
            print("\n🟢 Decoded model predictions:")
            for i, text in enumerate(decoded_preds):
                print(f"  [{i}] {text}")

            # Attention masks
            print("\n🩶 Attention masks (1=real token, 0=padding):")
            for i, mask in enumerate(attention_mask.tolist()):
                print(f"  [{i}] {mask}")

            # Decoded input IDs
            decoded_inputs = [
                self.processing_class.decode(seq, skip_special_tokens=False)
                for seq in input_ids
            ]
            print("\n🟦 Decoded input IDs:")
            for i, text in enumerate(decoded_inputs):
                print(f"  [{i}] {text}")

            # Decoded label IDs (with -100 replaced)
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = self.processing_class.pad_token_id
            decoded_labels_full = [
                self.processing_class.decode(lbl_seq.tolist(), skip_special_tokens=False)
                for lbl_seq in labels_for_decode
            ]
            print("\n🟧 Decoded label IDs (full, -100 → PAD):")
            for i, text in enumerate(decoded_labels_full):
                print(f"  [{i}] {text}")

            # === DEBUG: Verify label masking ===
            print("\n🔍 Token-level label masking inspection:")
            for i in range(min(2, len(input_ids))):  # print up to 2 examples
                ids = input_ids[i].tolist()
                lbls = labels[i].tolist()
                mask = attention_mask[i].tolist()

                decoded_labels = []
                for token_id, lbl_id in zip(ids, lbls):
                    tok = self.processing_class.decode([token_id], skip_special_tokens=False)
                    if lbl_id == -100:
                        decoded_labels.append(f"[MASKED:{tok}]")
                    else:
                        decoded_labels.append(tok)

                print(f"\n— Example {i} —")
                print(f"🧾 Input: {self.processing_class.decode(ids, skip_special_tokens=False)}")
                print(f"🏷️ Labels (MASKED means ignored in loss):")
                print("".join(decoded_labels))
                print(f"🩶 Attention mask:\n{mask}")
                print("-" * 60)
            print("=" * 80 + "\n")


        return (original_loss, outputs) if return_outputs else original_loss
        # ! Disabled custom loss for now until we know new labels and token is correct
        # === STEP 2. Find "answer" tokens using attention mask (CORRECTED) ===
        # The answer token is the *last* token in the padded sequence.
        # This is more robust than argmax(non_masked) which can pick prompt tokens.
        answer_indices = attention_mask.sum(dim=1) - 1 # shape [B]

        # Ensure long dtype for gather (CRITICAL)
        answer_indices = answer_indices.long()  # shape [B]
        
        vocab_size = outputs.logits.size(-1)

        # Expand for vocab dimension to gather logits: [B, 1, V]
        index_for_gather = answer_indices.view(-1, 1, 1).expand(-1, 1, vocab_size)
        answer_logits = torch.gather(outputs.logits, 1, index_for_gather).squeeze(1)  # [B, V]

        # Gather correct labels for the answer token: [B, 1] -> [B]
        answer_labels = torch.gather(labels, 1, answer_indices.view(-1, 1)).squeeze(1)  # [B]


        # === CRITICAL FIX: LOGITS MASKING TO CONSTRAIN OUTPUT ===
        # This prevents the model from choosing noisy, high-frequency tokens like '\n', 'j', '7', etc., 
        # at the final classification step.
        all_class_ids = self.signal_ids + self.background_ids
        
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=answer_logits.device)
        mask[all_class_ids] = True 
        
        # Drive logits for all unallowed tokens to negative infinity
        answer_logits[:, ~mask] = -1e9 
        # =======================================================


        # === STEP 3. Compute Focal Loss just on answer tokens (UNCHANGED) ===
        log_probs = F.log_softmax(answer_logits, dim=-1)
        P_t_log = torch.gather(log_probs, -1, answer_labels.unsqueeze(-1)).squeeze(-1)
        P_t = torch.exp(P_t_log)
        focal_term = ((1.0 - P_t).clamp(min=1e-6)) ** self.gamma

        # Class weights: emphasize signal (positive) samples
        weights = torch.gather(self.loss_weights, dim=0, index=answer_labels)

        focal_loss = -weights * focal_term * P_t_log
        focal_loss = focal_loss.mean()

        # === STEP 4. Blend with original loss (UNCHANGED) ===
        final_loss = original_loss + 0.2 * focal_loss

        # === STEP 5. DEBUGGING — Verify we're capturing correct answer positions ===
        with torch.no_grad():
            decoded_labels = [self.processing_class.decode([lbl.item()]) for lbl in answer_labels]
            pred_ids = torch.argmax(answer_logits, dim=-1)
            decoded_preds = [self.processing_class.decode([pid.item()]) for pid in pred_ids]

            print("=== [DEBUG: Answer Token Check] ===")
            print("Answer token indices:", answer_indices.tolist())
            print("Answer labels (token IDs):", answer_labels.tolist())
            print("Answer labels (decoded):", decoded_labels)
            print("Predicted IDs:", pred_ids.tolist())
            print("Predicted tokens (decoded):", decoded_preds)
            print("=====================================")


        return (final_loss, outputs) if return_outputs else final_loss
    
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
            self.model.numeric_fusion_adapter.load_state_dict(adapter_state_dict, strict=False)
            print("Successfully loaded custom numeric adapter weights.")
        else:
            print(f"Custom numeric adapter weights not found at {adapter_path}. Starting from scratch/random initialization.")

"""
Detailed Logger
"""
class GradLoggerCallback(TrainerCallback):
    def __init__(self, trainer):
        super().__init__()

        self.trainer = trainer

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        log_dict = {}

        if state.global_step % 16 != 0:
            # print("Skipping grad logging at step", state.global_step, state.global_step % 16)
            return
        
        adapter = model.numeric_fusion_adapter
        for name, param in adapter.named_parameters():
            if "weight" in name:
                log_dict.update({
                    f"adapter_{name.replace('.', '_')}_mean": param.data.mean().item(),
                    f"adapter_{name.replace('.', '_')}_std": param.data.std().item()
                })

            if param.grad is not None:
                log_dict.update({
                    f"adapter_{name.replace('.', '_')}_grad_mean": param.grad.mean().item(),
                    f"adapter_{name.replace('.', '_')}_grad_std": param.grad.std().item()
                })

        grads = [p.grad.norm(2) for p in adapter.parameters() if p.grad is not None]
        if len(grads) > 0:
            total_norm = torch.norm(torch.stack(grads), 2).item()
            log_dict.update({"adapter_grad_norm": total_norm})
        else:
            total_norm = 0.0
            log_dict.update({"adapter_grad_norm": total_norm})

        if log_dict:
            control.should_log = True
            self.trainer.log(log_dict)


"""
Training with NumericTrainer
"""
trainer = NumericTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    optimizers=(optimizer, None),  # Optimizer and no scheduler
    data_collator=NumericFeatureCollator(),
    # compute_loss_func=compute_loss  # our custom loss function
)

# Add detailed gradient logger callback
trainer.add_callback(GradLoggerCallback(trainer))

trainer.train(
    resume_from_checkpoint=not args.override_checkpoints
)
