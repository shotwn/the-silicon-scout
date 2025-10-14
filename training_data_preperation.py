"""
Run this after rd_data_processing.py to prepare the training data.
Input file: output/background_jets.jsonl and output/signal_jets.jsonl
Output files: output/train.jsonl and output/val.jsonl

Expected input file format (JSON lines):
{"jets": [{"px": 1403.5763346800975, "py": -674.5511371175991, "pz": -451.67074189253435, "E": 1638.7940107293846}, {"px": -1467.244438521324, "py": 611.5017730099569, "pz": 511.10171142099244, "E": 1670.1732010159512}], "type": "background", "num_particles": 231}
"""
import json
import random

# Open the input files, merge and shuffle the jets, then split into train and val sets
def load_jets(file_path, label):
    jets = []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            line = line.strip() # Remove any leading/trailing whitespace
            if not line: continue

            try:
                event = json.loads(line)
                event["type"] = label
                jets.append(event)
            except json.JSONDecodeError as e:
                print(f"JSON error on line {i}: {e}")
                print("Line bytes:", list(line.encode("utf-8")))
                break
    return jets

def merge_and_shuffle_jets(background_jets, signal_jets):
    all_jets = background_jets + signal_jets
    random.shuffle(all_jets)
    return all_jets

def split_jets(jets, train_ratio=0.8):
    train_size = int(len(jets) * train_ratio)
    train_jets = jets[:train_size]
    val_jets = jets[train_size:]
    return train_jets, val_jets

if __name__ == "__main__":
    background_jets = load_jets("output/background_jets.jsonl", "background")
    signal_jets = load_jets("output/signal_jets.jsonl", "signal")

    all_jets = merge_and_shuffle_jets(background_jets, signal_jets)

    train_jets, val_jets = split_jets(all_jets, train_ratio=0.8)

    # Save to output files
    with open("output/train.jsonl", "w") as f:
        for event in train_jets:
            f.write(json.dumps(event) + "\n")

    with open("output/val.jsonl", "w") as f:
        for event in val_jets:
            f.write(json.dumps(event) + "\n")