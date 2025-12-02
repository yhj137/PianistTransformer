import json
import os
import random

from miditoolkit import MidiFile
import pandas as pd
from tqdm import tqdm

from src.utils.midi import align_score_and_performance, ids_to_midi
from src.model.pianoformer import PianoT5GemmaConfig

if __name__ == "__main__":
    config = PianoT5GemmaConfig()
    data_path = "data/midis/asap-dataset-master/"
    metadata = pd.read_csv(os.path.join(data_path, "metadata.csv"))

    if not os.path.exists("temp"):
        os.makedirs("temp")
    with open("data/processed/sft/sft.jsonl", "w") as f:
        pass
    
    scores_set = set()
    for i in tqdm(range(len(metadata))):
        scores_set.add(metadata["midi_score"][i])
    
    random.seed(42)
    scores_set = sorted(list(scores_set))
    random.shuffle(scores_set)

    test_set = scores_set[:int(0.1 * len(scores_set))]

    print(test_set)

    data = []
    for i in tqdm(range(len(metadata))):
        split = "test" if metadata["midi_score"][i] in test_set else "train"
        score_midi_obj = MidiFile(os.path.join(data_path, metadata["midi_score"][i]))
        performance_midi_obj = MidiFile(os.path.join(data_path, metadata["midi_performance"][i]))
        try:
            xs, labels = align_score_and_performance(config, score_midi_obj, performance_midi_obj)
        except Exception as e:
            print(e)
            continue
        for j in range(len(xs)):
            data.append({
                "x": xs[j], 
                "label": labels[j], 
                "score_source": metadata["midi_score"][i], 
                "performance_source": metadata["midi_performance"][i], 
                "cut": j,
                "split": split
            })
            #ids_to_midi(config, xs[j]).dump(f"data/midis/sft_test/scores/{i}-{j}.mid")
            #ids_to_midi(config, labels[j]).dump(f"data/midis/sft_test/labels/{i}-{j}.mid")

            with open("data/processed/sft/sft.jsonl", "a") as f:
                #for i in data:
                f.write(json.dumps(data[-1])+"\n")
        print(f"Sample {i} successfully write!")
