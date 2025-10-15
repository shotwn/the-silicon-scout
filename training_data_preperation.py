"""
Run this after rd_data_processing.py to prepare the training data.
Input file: output/background_events.jsonl and output/signal_events.jsonl
Output files: output/train.jsonl and output/val.jsonl

Expected input file format (JSON lines):
{"jets": [{"px": 1403.5763346800975, "py": -674.5511371175991, "pz": -451.67074189253435, "E": 1638.7940107293846}, {"px": -1467.244438521324, "py": 611.5017730099569, "pz": 511.10171142099244, "E": 1670.1732010159512}], "type": "background", "num_particles": 231}
"""
import json
import random

# Open the input files, merge and shuffle the jets, then split into train and val sets
def load_events(file_path, label):
    events = []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            line = line.strip() # Remove any leading/trailing whitespace
            if not line: continue

            try:
                event = json.loads(line)
                event["type"] = label
                events.append(event)
            except json.JSONDecodeError as e:
                print(f"JSON error on line {i}: {e}")
                print("Line bytes:", list(line.encode("utf-8")))
                break
    return events

def merge_and_shuffle_events(background_events, signal_events):
    all_events = background_events + signal_events
    random.shuffle(all_events)
    return all_events

def split_events(events, train_ratio=0.8):
    train_size = int(len(events) * train_ratio)
    train_events = events[:train_size]
    val_events = events[train_size:]
    return train_events, val_events

if __name__ == "__main__":
    background_events = load_events("output/background_events.jsonl", "background")
    signal_events = load_events("output/signal_events.jsonl", "signal")

    # Addition
    # We want 1:1 ratio of background to signal in training/validation
    min_len = min(len(background_events), len(signal_events))

    all_events = merge_and_shuffle_events(background_events[:min_len], signal_events[:min_len])

    train_events, val_events = split_events(all_events, train_ratio=0.8)

    # Save to output files
    with open("output/train.jsonl", "w") as f:
        for event in train_events:
            f.write(json.dumps(event) + "\n")

    with open("output/val.jsonl", "w") as f:
        for event in val_events:
            f.write(json.dumps(event) + "\n")