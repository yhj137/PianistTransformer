from src.model.generate import batch_performance_render, map_midi
from src.model.pianoformer import PianoT5Gemma, PianoT5GemmaConfig
import torch
from datasets import load_dataset
import os
from miditoolkit import MidiFile
from src.utils.midi import midi_to_ids, ids_to_midi
import random

if __name__ == "__main__":    
    model = PianoT5Gemma.from_pretrained(
        "models/sft/",
        torch_dtype=torch.bfloat16
    )#.cuda()

    midis = []
    for i in range(1):
        midis.append(MidiFile(f"data/midis/testset/score/{i}.mid"))

    res = batch_performance_render(
        model, 
        midis, 
        temperature=1.0,
        top_p=0.95,
        device="cpu"
    )

    if not os.path.exists("data/midis/testset/inference"):
        os.makedirs("data/midis/testset/inference")

    for i, mid in enumerate(res):
        mid = map_midi(midis[i], mid)
        mid.dump(f"data/midis/testset/inference/{i}.mid")
