import os
from pathlib import Path
from miditoolkit import MidiFile
from src.utils.midi import normalize_midi

if __name__ == "__main__":
    path1 = Path("data/midis/testset/score")
    path2 = Path("data/midis/testset/human")
    path3 = Path("data/midis/testset/performance")
    files1 = list(path1.glob("*.mid"))
    files2 = list(path2.glob("*.mid"))
    files3 = list(path3.glob("*.mid"))

    if not os.path.exists(str(path1).replace("testset", "testset-norm")):
        os.makedirs(str(path1).replace("testset", "testset-norm"))
    for i in files1:
        i = str(i)
        normalize_midi(MidiFile(i)).dump(i.replace("testset", "testset-norm"))

    if not os.path.exists(str(path2).replace("testset", "testset-norm")):
        os.makedirs(str(path2).replace("testset", "testset-norm"))
    for i in files2:
        i = str(i)
        normalize_midi(MidiFile(i)).dump(i.replace("testset", "testset-norm"))

    if not os.path.exists(str(path3).replace("testset", "testset-norm")):
        os.makedirs(str(path3).replace("testset", "testset-norm"))
    for i in files3:
        i = str(i)
        normalize_midi(MidiFile(i)).dump(i.replace("testset", "testset-norm"))
