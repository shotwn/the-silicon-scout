from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, TrainingArguments, Trainer
from datasets import load_dataset
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

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

lora_config = LoraConfig(
    r=12,
    lora_alpha=48,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.15,
    bias="none"
)

model = get_peft_model(model, lora_config)

# Load dataset
ds = load_dataset("json", data_files={"train":"output/train.jsonl", "validation":"output/val.jsonl"})

def format_example(example):
    jets = example["jets"]
    s = "[INST] Classify this event as 'signal' or 'background'.\n"
    s += "jets:\n"
    for i, j in enumerate(jets):
        s += f"  jet{i+1}: px={j['px']:.10f} py={j['py']:.10f} pz={j['pz']:.10f} E={j['E']:.10f}\n"
    s += f"num_particles: {example['num_particles']}[/INST]"

    # HF Trainer expects 'labels'
    return {"input_text": s, "labels": example["type"]}


ds = ds.map(format_example)

tokenizer.pad_token = tokenizer.eos_token
def tokenize_example(example):
    # Tokenize the input text
    encoding = tokenizer(
        example["input_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    
    # Tokenize label as a single token
    label_id = tokenizer(example["labels"], add_special_tokens=False)["input_ids"][0]
    
    # Initialize labels to -100 (ignore)
    labels = [-100] * 256
    # Put the label token at the last position (or wherever you want)
    labels[-1] = label_id  # could be [0] or [input_length - 1]

    # Assign *only* the input_ids from the label tokenization
    encoding["labels"] = labels
    return encoding

def tokenize_example_refined(example, max_length=256):
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
    return full_encoding

tokenized_ds = ds.map(tokenize_example_refined, batched=False)

# Heck if I know, it was needed to make max_grad_norm work
model.enable_input_require_grads()

# Setup training
training_args = TrainingArguments(
    output_dir="lora_lhco",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16, # simulate larger batch size
    learning_rate=3e-5, # Reduced from 2e-4 to 3e-5 after seeing loss spikes in initial checkpoints
    num_train_epochs=5,
    save_strategy="steps",
    save_steps=100,
    eval_steps=100,
    logging_steps=50,
    gradient_checkpointing=True, # Transformer was warning me to set this.
    gradient_checkpointing_kwargs={'use_reentrant': False},  # Transformer was warning me to set this. Appearently in future versions it will default to False.
    max_grad_norm=1.0 # Added gradient clipping, after custom weighted loss we were having huge grad_norms (>500) in initial checkpoints
)

# Use 8-bit Adam optimizer
from bitsandbytes.optim import AdamW8bit
optimizer = AdamW8bit(model.parameters(), lr=training_args.learning_rate)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    optimizers=(optimizer, None),  # use custom 8-bit optimizer
    # compute_loss_func=compute_loss  # our custom loss function
)

trainer.train(
    resume_from_checkpoint=True
)
