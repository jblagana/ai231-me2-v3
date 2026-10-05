# V2 sanity: slots selftest + dataset init on real raw_v2 data (CPU-only)
import sys
sys.path.insert(0, "src")
from slots import selftest, extract_slot
assert selftest(), "slots selftest FAILED"
print("spot play a song:", extract_slot("play_music", "play a song"))
print("spot turn off the music:", extract_slot("play_music", "turn off the music"))
print("spot dim to 50:", extract_slot("dim_lights", "dim the lights to fifty percent"))
from pathlib import Path
from train_v2 import VCMDatasetV2
for split in ("train", "eval"):
    ds = VCMDatasetV2(Path("/home/jan.rhey.lagana/vcm/data/raw_v2"), split)
    nslot = sum(1 for it in ds.items if it[3] >= 0)
    print(split, "n=", len(ds), "slot_items=", nslot)
print("DATASET-OK")
