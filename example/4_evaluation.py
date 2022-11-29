import sys
import os
import json

from IPython import embed
from src.Dataset import Dataset

with open(sys.argv[1], 'r') as f:
  target_configs = json.load(f)

dataset = Dataset(target_configs,sys.argv[1])
results_path = os.path.join(sys.argv[2], 'detections')

print("Evaluation begin.")
evaluation = dataset.evaluate_test_images(results_path)
print("Evaluation completed!")

print(json.dumps(evaluation, indent = 3))

with open(f"{sys.argv[2]}/evaluation.json", 'w') as f:
  json.dump(evaluation, f)
