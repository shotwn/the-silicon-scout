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
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)

# Load dataset
ds = load_dataset("json", data_files={"train":"output/train.jsonl", "validation":"output/val.jsonl"})

def format_example(example):
    jets = example["jets"]
    s = "jets:\n"
    for i, j in enumerate(jets):
        s += f"  jet{i+1}: px={j['px']:.10f} py={j['py']:.10f} pz={j['pz']:.10f} E={j['E']:.10f}\n"
    s += f"num_particles: {example['num_particles']}\n"
    s += "Output:"

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
    
    # Tokenize the label text
    labels_encoding = tokenizer(
        example["labels"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

    # Assign *only* the input_ids from the label tokenization
    encoding["labels"] = labels_encoding["input_ids"]
    return encoding

tokenized_ds = ds.map(tokenize_example, batched=False)

# Heck if I know, it was needed to make max_grad_norm work
model.enable_input_require_grads()
# Setup training
training_args = TrainingArguments(
    output_dir="lora_lhco",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4, # Reduced from 2e-4 to 1e-4 after seeing loss spikes in initial checkpoints
    num_train_epochs=5,
    save_strategy="steps",
    save_steps=100,
    eval_steps=100,
    logging_steps=50,
    gradient_checkpointing=True, # Transformer was warning me to set this.
    gradient_checkpointing_kwargs={'use_reentrant': False},  # Transformer was warning me to set this. Appearently in future versions it will default to False.
    max_grad_norm=1.0 # Added gradient clipping, after custom weighted loss we were having huge grad_norms (>500) in initial checkpoints
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    # compute_loss_func=compute_loss  # our custom loss function
)

trainer.train(
    resume_from_checkpoint=True
)
