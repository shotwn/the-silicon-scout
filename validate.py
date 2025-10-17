from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import PeftModel
import torch
import json
from tqdm import tqdm
import os
import glob
from argparse import ArgumentParser

device = "cuda" if torch.cuda.is_available() else "cpu"

base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
lora_checkpoint = "lora_lhco/checkpoint-*"  # path to LoRA weights

# Check environment
arg_parser = ArgumentParser()
arg_parser.add_argument("--use_checkpoint", type=int, default=0, required=False, help="Give custom checkpoint number to use, or 0 for latest.")
args = arg_parser.parse_args()

# Get the latest folder in the checkpoint directory
list_of_dirs = glob.glob(lora_checkpoint)
lora_checkpoint = max(list_of_dirs, key=os.path.getctime)

if args.use_checkpoint != 0:
    lora_checkpoint = f"lora_lhco/checkpoint-{args.use_checkpoint}"

print("Using LoRA checkpoint:", lora_checkpoint)

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
    jets = example["jets"]
    s = "[INST] Classify this event as 'signal' or 'background'.\n"
    s += "jets:\n"
    for i, j in enumerate(jets):
        s += f"  jet{i+1}: P_T={j['P_T']:.10f} eta={j['eta']:.10f} phi={j['phi']:.10f} E={j['E']:.10f} m={j['m']:.10f} n_particles={j['n_particles']} P_T_lead={j['P_T_lead']:.10f}\n"
        for dR_jet, dR_value in j["dR"].items():
            dR_value = dR_value if dR_value is not None else 0.0
            s += f"    dR_{dR_jet}={dR_value:.2f}\n"
    s += f"n_particles: {example['n_particles']} M_jj= {example['M_jj']}[/INST]"

    return s

preds = []
labels = []
target_names = ["background", "signal"]
for example in tqdm(val_examples):
    prompt = make_prompt(example)
    # Encode with attention_mask and padding
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

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
    print("Expected output:")
    print(example["type"])
    print("---\n\n")


    # Decode output tokens after prompt
    pred_text = tokenizer.decode(
        output_ids[0][inputs['input_ids'].shape[1]:],  # only new tokens
        skip_special_tokens=True,
    )

    # Take only the last word as prediction
    # We will split the pred_text in a way to take away the prompt part
    if pred_text:
        pass
        #pred_text = pred_text[inputs['input_ids'].shape[1]:].strip().lower().split()[0]

    # print(f"Prompt:\n{prompt}\nPrediction: '{pred_text}'\nTrue label: '{example['type']}'\nResult: {'CORRECT' if pred_text == example['type'].lower() else 'WRONG'}\n---")
    
    # Normalize output
    if "signal" in pred_text:
        pred = "signal"
    elif "background" in pred_text:
        pred = "background"
    else:
        # fallback if model generates something else
        pred = "unknown"
        if "unknown" not in target_names:
            target_names.append("unknown")
    
    preds.append(pred)
    labels.append(example["type"].lower())


# Simple accuracy
accuracy = sum(p == l for p, l in zip(preds, labels)) / len(labels)
print("Validation Accuracy:", accuracy)

# Print empty predictions
num_unknown = sum(1 for p in preds if p == "unknown")
print("Number of unknown predictions:", num_unknown)

# Optional: more metrics with sklearn
from sklearn.metrics import classification_report
print(classification_report(labels, preds, target_names=target_names))