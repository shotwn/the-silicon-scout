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
    0: "7GB",
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
        s += f"  jet{i+1}: px={j['px']:.5f} py={j['py']:.5f} pz={j['pz']:.5f} E={j['E']:.5f}\n"
    s += f"num_particles: {example['num_particles']}\n"
    s += "Output:"

    # HF Trainer expects 'labels'
    return {"input_text": s, "labels": 1 if example["type"] == "signal" else 0}


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
    
    # Convert label to integer
    label_int = 1 if example["labels"] == 1 else 0  # already 1/0 in your format_example

    # Create labels array same length as input_ids
    # 100 is a special token to ignore in loss computation
    labels = [-100] * len(encoding["input_ids"])  # ignore all tokens by default
    labels[-1] = label_int  # put the numeric label at the last token position

    encoding["labels"] = labels
    return encoding

tokenized_ds = ds.map(tokenize_example, batched=False)

# Setup training
training_args = TrainingArguments(
    output_dir="lora_lhco",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=4,
    save_strategy="steps",
    save_steps=100,
    eval_steps=100,
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
)

trainer.train()
