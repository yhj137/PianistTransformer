import json
import os

from miditoolkit import MidiFile
import pandas as pd
from tqdm import tqdm

from src.utils.midi import midi_to_ids
from src.model.pianoformer import PianoT5GemmaConfig

if __name__ == "__main__":
    with open("data/midis/aria-midi-v1-ext/metadata.json") as f:
        metadata = json.load(f)
    filter_list = []
    for k1 in metadata.keys():
        for k2 in metadata[k1]["audio_scores"].keys():
            if metadata[k1]["audio_scores"][k2] > 0.95:
                try:
                    filter_list.append(("{:0>{}}".format(k1, 6)+f"_{k2}.mid", metadata[k1]["metadata"]["genre"], int(k1)))
                except:
                    pass
    tongji = {}
    for i in filter_list:
        if i[1] not in tongji:
            tongji[i[1]] = 0
        tongji[i[1]] += 1
    print(tongji)
    print(len(filter_list))
    path_dir = []
    for i in range(97,123):
        for j in range(97,123):
            path_dir.append(f"data/midis/aria-midi-v1-ext/data/{chr(i)+chr(j)}/")

    config = PianoT5GemmaConfig()
    output = []
    cnt = 0
    for i in tqdm(filter_list):
        try:
            genre = i[1]
            file_name = i[0]
            number = i[2] - 1
            dir_name = path_dir[number // 2000]
            real_file_name= dir_name + file_name
            midi_obj = MidiFile(real_file_name)
            ids = midi_to_ids(config, midi_obj)
            output.append({"input_ids": ids, "source": file_name, "genre": genre})
            if len(output) >= 1000:
                with open(f"data/processed/pretrain/raw/aria/{cnt}.jsonl", "w") as f:
                    for j in output:
                        f.write(json.dumps(j)+"\n")
                cnt += 1
                output = []
        except:
            pass
    with open(f"data/processed/pretrain/raw/aria/{cnt}.jsonl", "w") as f:
        for j in output:
            f.write(json.dumps(j)+"\n")
        cnt += 1
        output = []
    