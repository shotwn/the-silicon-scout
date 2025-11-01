from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import PeftModel
import torch
import json
from tqdm import tqdm
import os
import glob
from argparse import ArgumentParser
from numeric_fusion_adapter import NumericFusionAdapter

device = "cuda" if torch.cuda.is_available() else "cpu"

base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
lora_checkpoint = "lora_lhco/checkpoint-*"  # path to LoRA weights

# Check environment
arg_parser = ArgumentParser()
arg_parser.add_argument("--use_checkpoint", type=int, default=0, required=False, help="Give custom checkpoint number to use, or 0 for latest.")
arg_parser.add_argument("--validation_dataset", type=str, default="output/val_one_to_one.jsonl", required=False, help="Path to validation dataset.")
arg_parser.add_argument("--sample_size", type=int, default=1000, required=False, help="Number of samples to use from validation dataset for quick testing. Default is 1000. Use 0 to use all dataset.")
arg_parser.add_argument("--show_generated", action="store_true", help="If set, will show the full generated text from the model.")
arg_parser.add_argument("--use_numeric", action="store_true", help="If set, will use numeric features in the model.")
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

# Initialize and Load Numeric Fusion Adapter
numeric_dim = 16 
hidden_size = model.config.hidden_size
model.numeric_fusion_adapter = NumericFusionAdapter(hidden_size, numeric_dim, dtype=torch.float16, device=device).to(device)

# Load the trained adapter weights
adapter_weights_path = os.path.join(lora_checkpoint, "numeric_fusion_adapter.bin")
if args.use_numeric:
    try:
        adapter_state_dict = torch.load(adapter_weights_path, map_location=device)
        model.numeric_fusion_adapter.load_state_dict(adapter_state_dict)
        print(f"Loaded NumericFusionAdapter weights from {adapter_weights_path}")
    except FileNotFoundError:
        print(f"WARNING: NumericFusionAdapter weights not found at {adapter_weights_path}. Adapter is using random weights!")

# Apply LoRA weights
model = PeftModel.from_pretrained(
    model,
    lora_checkpoint,
    device_map="auto"
)

# Optionally merge the LoRA
# Merge LoRA weights into base model
# model = model.merge_and_unload()

model.to(device)
model.eval()

print("Model and tokenizer loaded.")

# Load validation dataset
val_file = args.validation_dataset
val_examples = []

with open(val_file, "r") as f:
    i = 0
    limit_to = args.sample_size  # limit for quick testing; set to None to use all
    for line in f:
        val_examples.append(json.loads(line))
        i += 1
        if limit_to is not None and limit_to != 0 and i >= limit_to:
            break


def make_prompt(example):
    """Create prompt from example."""
    jets = example["jets"]
    s = "[INST] Classify this event as 'signal' or 'background'.\n"
    s += "jets:\n"
    for i, j in enumerate(jets):
        s += f"  jet{i+1}: P_T={j['P_T']:.10f} eta={j['eta']:.10f} phi={j['phi']:.10f} E={j['E']:.10f} m={j['m']:.10f} n_particles={j['n_particles']} P_T_lead={j['P_T_lead']:.10f}\n"
        for dR_jet, dR_value in j["dR"].items():
            dR_value = dR_value if dR_value is not None else 0.0
            s += f"    dR_{dR_jet}={dR_value:.2f}\n"
    s += f"n_particles: {example['n_particles']} M_jj= {example['M_jj']}[/INST]\n"

    return s

def extract_numeric_features(example):
    """Extract numeric features from example."""
    jets = example["jets"]
    numeric_vector = [0.0] * 16
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

    return torch.tensor(numeric_vector, dtype=torch.float16).unsqueeze(0).to(device)

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

    # Inject numeric embedding only if requested
    if args.use_numeric:
        # Get numeric features and compute embeddings
        numeric_features = extract_numeric_features(example)
        # Get numeric embeddings
        numeric_embeds = model.numeric_fusion_adapter(numeric_features)
        # Get text embeddings
        text_embeds = model.get_input_embeddings()(input_ids)
        # Combine embeddings and adjust attention mask
        inputs_embeds = torch.cat([numeric_embeds, text_embeds], dim=1)
        # Adjust attention mask
        attention_mask_adjusted = torch.cat(
            [torch.ones((attention_mask.shape[0], 1), device=device), attention_mask],
            dim=1
        )

        # Prepare generation kwargs
        gen_kwargs = dict(inputs_embeds=inputs_embeds, attention_mask=attention_mask_adjusted)

        # The output decoding index needs to be adjusted in the next block!
        # start_decode_index = input_ids.shape[1] + 1 # +1 for the numeric token
        start_decode_index = 0 # ^ This one was wrong, since we are using input_embeds we just get the answer part. No prompt to offset. 
                               # Also first token is not there in the text output.

        # print("Numeric embed mean/std:", numeric_embeds.mean().item(), numeric_embeds.std().item())
        # print("Text embed mean/std:", text_embeds.mean().item(), text_embeds.std().item())
    else:
        gen_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        start_decode_index = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **gen_kwargs,
            #top_p=0.9,                     # nucleus sampling ?
            max_new_tokens=1,              # allow at least 2 new tokens
            temperature=0.01,               # almost deterministic
            #do_sample=True,               # disable greedy decoding
            do_sample=False,               # enable greedy decoding -> reduce randomness
            eos_token_id=tokenizer.eos_token_id,  # 🚫 prevent early stopping
            suppress_tokens=[tokenizer.eos_token_id],  # 🚫 block EOS generation
            pad_token_id=tokenizer.eos_token_id # suppress warning that appears when this is set automatically
        )

    if args.show_generated:
        print("===")
        print(f"Input length: {inputs['input_ids'].shape[1]}, Output length: {output_ids.shape[1]}")
        print("Raw model output:")
        for i in range(len(output_ids)):
            print(tokenizer.decode(output_ids[i], skip_special_tokens=False))
        print("---")
        print("Expected output:")
        print(example["type"])
        print("---\n\n")


    # Decode output tokens after prompt
    pred_text = tokenizer.decode(
        output_ids[0][start_decode_index:],  # Use the calculated index, takes only newly generated tokens
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

# Results
# Number of background, signal or unknown predictions with correct labels
for target in target_names:
    num_total = labels.count(target)
    num_correct = sum(1 for p, l in zip(preds, labels) if p == target and l == target)
    print(f'Number of correct {target} predictions: {num_correct} out of {num_total}')

# Simple accuracy
accuracy = sum(p == l for p, l in zip(preds, labels)) / len(labels)
print("Validation Accuracy:", accuracy)

# Print empty predictions
num_unknown = sum(1 for p in preds if p == "unknown")
if num_unknown > 0:
    print("WARNING: Number of unknown predictions:", num_unknown)
else:
    print("All predictions classified as 'signal' or 'background'.")

# Optional: more metrics with sklearn
from sklearn.metrics import classification_report
print(classification_report(labels, preds, target_names=target_names))

# Added: SIC calculation
background_total = labels.count("background")
signal_total = labels.count("signal")
signal_selected = sum(1 for p, l in zip(preds, labels) if p == "signal" and l == "signal")

# Fraction of true signal events correctly identified (epsilon_s)
signal_efficiency = signal_selected / signal_total if signal_total > 0 else 0.0

# Fraction of background events incorrectly identified as signal (epsilon_b)
background_selected = sum(1 for p, l in zip(preds, labels) if p == "signal" and l == "background")
background_efficiency = background_selected / background_total if background_total > 0 else 0.0

# Significance Improvement Characteristic (SIC)
if background_efficiency > 0:
    sic = signal_efficiency / (background_efficiency ** 0.5)
    print(f"SIC (Significance Improvement Characteristic): {sic:.4f}")
else:
    print("SIC (Significance Improvement Characteristic): Undefined (no background events selected as signal)")


