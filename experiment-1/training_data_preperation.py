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

def shuffle_events(events):
    random.shuffle(events)
    return events

def merge_events(background_events, signal_events):
    return background_events + signal_events

def split_events(events, train_ratio=0.8):
    train_size = int(len(events) * train_ratio)
    train_events = events[:train_size]
    val_events = events[train_size:]
    return train_events, val_events

if __name__ == "__main__":
    background_events = load_events("output/background_events.jsonl", "background")
    signal_events = load_events("output/signal_events.jsonl", "signal")

    # Addition


    # We will generate 2 sets. One with equal number of signal/background, one with original ratio
    # We will make sure one set doesn't have validation events inside other set's training events

    all_events_original_ratio = merge_events(background_events, signal_events)
    all_events_original_ratio = shuffle_events(all_events_original_ratio)

    train_events_original_ratio, val_events_original_ratio = split_events(all_events_original_ratio, train_ratio=0.8)

    with open("output/train_original_ratio.jsonl", "w") as f:
        for event in train_events_original_ratio:
            f.write(json.dumps(event) + "\n")

    with open("output/val_original_ratio.jsonl", "w") as f:
        for event in val_events_original_ratio:
            f.write(json.dumps(event) + "\n")
  

    # To keep validation and training sets disjoint, we will use original ratio validation and training sets
    # but pick equal number of signal/background for training set only from training set
    # and for validation set only from validation set

    # Find the minimum count between signal and background in training set
    train_background = [e for e in train_events_original_ratio if e["type"] == "background"]
    train_signal = [e for e in train_events_original_ratio if e["type"] == "signal"]

    # Find the minimum count between signal and background in validation set
    # So we do use all of the small set and pick equal number from the larger set
    min_train_len = min(len(train_background), len(train_signal))
    train_events = shuffle_events(merge_events(train_background[:min_train_len], train_signal[:min_train_len]))

    # Similarly for validation set
    val_background = [e for e in val_events_original_ratio if e["type"] == "background"]
    val_signal = [e for e in val_events_original_ratio if e["type"] == "signal"]
    min_val_len = min(len(val_background), len(val_signal))
    val_events = shuffle_events(merge_events(val_background[:min_val_len], val_signal[:min_val_len]))

    # Note that shuffling again is fine since we already split into train/val sets

    # Save to output files
    with open("output/train_one_to_one.jsonl", "w") as f:
        for event in train_events:
            f.write(json.dumps(event) + "\n")

    with open("output/val_one_to_one.jsonl", "w") as f:
        for event in val_events:
            f.write(json.dumps(event) + "\n")


    print(f"Prepared training data: {len(train_events)} training events, {len(val_events)} validation events.")
    print(f"Prepared training data (original ratio): {len(train_events_original_ratio)} training events, {len(val_events_original_ratio)} validation events.")
