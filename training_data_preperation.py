"""
Run this after rd_data_processing.py to prepare the training data.
Input file: output/background_events.jsonl and output/signal_events.jsonl
Output files: output/train.jsonl and output/val.jsonl

Expected input file format (JSON lines):
{"type": "background", "jets": [{"P_T": 1204.8690323851608, "eta": 0.1861334289679372, "phi": -2.357433740664133, "E": 1249.3582717067266, "m": 241.46999128671604, "n_particles": 76, "P_T_lead": 167.34600830078125, "dR": {"jet2": 3.1309376303974816}}, {"P_T": 1252.3324471178576, "eta": 0.09829797821454976, "phi": 0.7722715775774975, "E": 1259.8006950853248, "m": 59.65166757698319, "n_particles": 44, "P_T_lead": 323.1458435058594, "dR": {"jet1": 3.1309376303974816}}], "num_particles": 164, "M_jj": 301.1216588636992}
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

    # Add new pairs without equal number of signal/background
    all_events_original_ratio = merge_and_shuffle_events(background_events, signal_events)
    train_events_original_ratio, val_events_original_ratio = split_events(all_events_original_ratio, train_ratio=0.8)
    with open("output/train_original_ratio.jsonl", "w") as f:
        for event in train_events_original_ratio:
            f.write(json.dumps(event) + "\n")

    with open("output/val_original_ratio.jsonl", "w") as f:
        for event in val_events_original_ratio:
            f.write(json.dumps(event) + "\n")
            
    print(f"Prepared training data: {len(train_events)} training events, {len(val_events)} validation events.")
    print(f"Prepared training data (original ratio): {len(train_events_original_ratio)} training events, {len(val_events_original_ratio)} validation events.")
