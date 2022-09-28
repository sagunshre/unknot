import csv
import json
from IPython import embed

with open('metadata.csv', 'w') as meta_file, open('annotations/train.csv', 'r') as csv_file:
  data = csv.reader(csv_file, delimiter=",")
  fieldnames = ["filename","distance"]
  next(data, None)
  writer = csv.writer(meta_file)
  writer.writerow(fieldnames)
  for row in data:
    attr = json.loads(row[14])
    writer.writerow([row[8], float(attr['metadata']['area'])])