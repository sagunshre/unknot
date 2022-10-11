import sys
import os
import json
from src.Dataset import Dataset
from IPython import embed

with open(sys.argv[2], 'r') as f:
  target_configs = json.load(f)

with open(sys.argv[1], 'r') as f:
  source_configs = json.load(f)
  for source_config in source_configs:
    source = Dataset(source_config, sys.argv[1])
    target = Dataset(target_configs, sys.argv[2])

    # Get number of usable CPUs.
    workers = len(os.sched_getaffinity(0))

    annotation_patches = source.generate_annotation_patches(scale_transfer_target=target, max_workers=workers, reuse_patches=True)
