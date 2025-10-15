from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import PeftModel
import torch
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
lora_checkpoint = "lora_lhco/checkpoint-2000"  # path to LoRA weights

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token  # ensure padding

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # replaces torch_dtype
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name, 
    device_map="auto",
    quantization_config=bnb_config,  # replaces load_in_4bit argument
)

# Apply LoRA weights
model = PeftModel.from_pretrained(
    model,
    lora_checkpoint,
    device_map="auto"
)

# Merge LoRA weights into base model
# model = model.merge_and_unload()

model.to(device)
model.eval()

print("Model and tokenizer loaded.")


val_file = "output/val.jsonl"
val_examples = []

with open(val_file, "r") as f:
    i = 0
    limit_to = 1000  # limit for quick testing; set to None to use all
    for line in f:
        val_examples.append(json.loads(line))
        i += 1
        if limit_to is not None and i >= limit_to:
            break

def make_prompt(example):
    s = "jets:\n"
    for i, j in enumerate(example["jets"]):
        s += f"  jet{i+1}: px={j['px']:.10f} py={j['py']:.10f} pz={j['pz']:.10f} E={j['E']:.10f}\n"
    s += f"num_particles: {example['num_particles']}\nOutput:"
    return s

preds = []
labels = []

for example in val_examples:
    prompt = make_prompt(example)
    # Encode with attention_mask and padding
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Generate 2 tokens max (for "0"/"1" or "background"/"signal")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            top_p=0.9,                     # nucleus sampling ?
            max_new_tokens=1,              # allow at least 2 new tokens
            temperature=0.01,               # almost deterministic
            do_sample=True,               # disable greedy decoding
            eos_token_id=None,  # 🚫 prevent early stopping
            suppress_tokens=[tokenizer.eos_token_id],  # 🚫 block EOS generation
            attention_mask=attention_mask
        )


    print("===")
    print(f"Input length: {inputs['input_ids'].shape[1]}, Output length: {output_ids.shape[1]}")
    print("Raw model output:")
    print(tokenizer.decode(output_ids[0], skip_special_tokens=False))
    print("---")
    
    
    # Decode output tokens after prompt
    pred_text = tokenizer.decode(
        output_ids[0], 
        skip_special_tokens=True
    ).strip().lower()

    # Take only first word to avoid repeats
    if pred_text:
        pred_text = pred_text[len(prompt):].strip()  # take the first word after the prompt
        pred_text = pred_text.split(' ')[0]  # take only the first word

    #print(f"Prompt:\n{prompt}\nPrediction: '{pred_text}'\nTrue label: '{example['type']}'\nResult: {'CORRECT' if pred_text == example['type'].lower() else 'WRONG'}\n---")
    
    # Normalize output
    if "signal" in pred_text:
        pred = "signal"
    elif "background" in pred_text:
        pred = "background"
    else:
        # fallback if model generates something else
        pred = "null"
    
    preds.append(pred)
    labels.append(example["type"].lower())


# Simple accuracy
accuracy = sum(p == l for p, l in zip(preds, labels)) / len(labels)
print("Validation Accuracy:", accuracy)

# Optional: more metrics with sklearn
from sklearn.metrics import classification_report
print(classification_report(labels, preds, target_names=["background", "signal"]))