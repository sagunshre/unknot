import sys
import os
import json
from src.Dataset import Dataset
from src.ObjectDetection import ObjectDetector
from IPython import embed
import src.mrcnn.utils as utils


with open(sys.argv[2], 'r') as f:
  target_configs = json.load(f)

with open(sys.argv[1], 'r') as f:
  source_configs = json.load(f)

source = Dataset(source_configs, sys.argv[1])
target = Dataset(target_configs, sys.argv[2])
results = sys.argv[3]

annotation_patches = source.get_annotation_patches()

detector = ObjectDetector(os.path.join(results, 'model'))

config = {
   'RPN_NMS_THRESHOLD': 0.85,
   'IMAGES_PER_GPU': 5,
}

train_scheme = [
   {
      'layers': 'heads',
      'epochs': 10,
      'learning_rate': 0.001
   },
   {
      'layers': 'heads',
      'epochs': 10,
      'learning_rate': 0.0005
   },
   {
      'layers': 'heads',
      'epochs': 10,
      'learning_rate': 0.0001
   },
   {
      'layers': 'all',
      'epochs': 10,
      'learning_rate': 0.0001
   },
   {
      'layers': 'all',
      'epochs': 10,
      'learning_rate': 0.00005
   },
   {
      'layers': 'all',
      'epochs': 10,
      'learning_rate': 0.00001
   },
]

coco_model_path = 'mask_rcnn_coco.h5'
if not os.path.exists(coco_model_path):
   utils.download_trained_weights(coco_model_path)

detector.perform_inference(annotation_patches, target, os.path.join(results, 'detections'))
