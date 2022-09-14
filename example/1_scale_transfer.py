import sys
import os
import csv

from IPython import embed


from src.Dataset import Dataset

source = Dataset(sys.argv[1])
target = Dataset(sys.argv[2])

# Get number of usable CPUs.
workers = len(os.sched_getaffinity(0))

annotation_patches = source.generate_annotation_patches(scale_transfer_target=target, max_workers=workers, reuse_patches=True)

og_images = list(source.get_train_images())

label_counter = {}

for image in og_images:
  for annotation in image.annotations:
    if not annotation.label_id in label_counter:
      label_counter[annotation.label_id] = 1
    else:
      label_counter[annotation.label_id] += 1

train_report = source.get_config_path('train_annotations_file')

total_patches = 0
with open(train_report, 'r') as file:
    reader = csv.reader(file)
    next(reader)
    total_patches = sum(1 for row in reader)


label_composition = {}
for label, count in label_counter.items():
  label_composition[label] = (count/total_patches) * 100

embed()

print("TEST")