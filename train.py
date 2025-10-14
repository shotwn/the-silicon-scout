from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, TrainingArguments, Trainer
from datasets import load_dataset

# Example: using a 7B model
model_name = "mistralai/mistral-7b-instruct"  # example name
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
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
ds = load_dataset("json", data_files={"train":"train.jsonl", "validation":"val.jsonl"})

def format_example(example):
    # turn your event JSON into prompt / input text
    jets = example["jets"]
    s = "jets:\n"
    for i, j in enumerate(jets):
        s += f"  jet{i+1}: px={j['px']:.2f} py={j['py']:.2f} pz={j['pz']:.2f} E={j['E']:.2f}\n"
    s += f"num_particles: {example['num_particles']}\n"
    s += "Output:"
    return {"input_text": s, "label": example["type"]}

ds = ds.map(format_example)

# Setup training
training_args = TrainingArguments(
    output_dir="lora_lhco",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_epochs=4,
    save_strategy="steps",
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
)

trainer.train()
