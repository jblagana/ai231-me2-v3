# Sanity: build wake datasets on real data (counts must match the manifest)
import sys
sys.path.insert(0, 'src')
from pathlib import Path
import train_wake as tw

W = Path('/home/jan.rhey.lagana/vcm/data/wake_v2')
C = Path('/home/jan.rhey.lagana/vcm/data/raw_v2')
N = Path('/home/jan.rhey.lagana/vcm/data/noise16k_raw')

d = tw.WakeDataset('train', W, C, N)
print('train items:', len(d),
      'wake:', sum(1 for _, y in d.items if y == 1))
e = tw.WakeDataset('eval', W, C)
print('eval items:', len(e),
      'wake:', sum(1 for _, y in e.items if y == 1))
