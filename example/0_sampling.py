import sys
import os
import csv
import yaml
import json
import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from IPython import embed

data = []
with open("../data/source_S155/annotations/train.csv", 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
      data.append(row)

data = np.array(data, dtype="U256")
counter =  Counter(data[:, 1])
total_patches = data.shape[0]
sorted_labels = Counter(dict(sorted(counter.items(), key=lambda pair: pair[1], reverse=True)))

with open('exp_config.yml', 'r') as file:
  experiments = yaml.unsafe_load(file)

# Balancing rule
for key, rule in enumerate(experiments["rules"]):
  name = rule['name']
  sample_sizes = dict(zip(list(sorted_labels.keys()), [0]*len(sorted_labels.keys())))

  for k, v in sorted_labels.items():
    label_maj = list(sorted_labels.values())[0]

    if(v == label_maj):
      sample_sizes[k] = int(round(label_maj))
      continue

    if(name == "rule-50"):
      if(v < label_maj/2):
        sample_sizes[k] = int(round(label_maj/2))
      else:
        sample_sizes[k] = int(round(sorted_labels[k]))

    if(name == "rule-75"):
      sample_sizes[k] = int(round((3 * label_maj)/4))

    if(name == "rule-50-75-100"):
      if(v <= label_maj/4):
        sample_sizes[k] = int(round((1/2) * label_maj))
      elif(v > label_maj/4 and v <= label_maj/2):
        sample_sizes[k] = int(round((3 * label_maj)/4))
      else:
        sample_sizes[k] = int(round(label_maj))

    if(name == "rule-50-100"):
      if(v >= label_maj/2):
        sample_sizes[k] = int(round(label_maj/2))
      else:
        sample_sizes[k] = int(round(label_maj))
  experiments["rules"][key]['sample_sizes'] = sample_sizes

with open('exp_config.yml', 'w') as exp_config:
  yaml.dump(json.loads(json.dumps(experiments)), exp_config)

src = dict(list(zip(list(range(0,len(data[:, 0].tolist()))), data[:, 0].tolist())))
X = np.array(list(src.keys())).reshape(-1, 1)
y = data[:, 1]

# SMOTE and random undersampling
for experiment in experiments["rules"]:
  csv_rows = []
  sampling_strategy = experiment["sample_sizes"]
  # check if any label needs to be under sampled
  combination = np.greater_equal(np.array(list(sampling_strategy.values())), np.array(list(sorted_labels.values())))
  if(np.any(combination==False)):
    # for oversampling and under sampling rule 50-100

    # Get X and y for over samplings and use SMOTE
    over_sampling_indices = np.array(np.where(combination==True)).flatten().tolist()
    over_sampling_strategy = dict([list(sampling_strategy.items())[i] for i in over_sampling_indices])
    over_data_indices = [list(np.array(np.where(data==[i])[0]).flatten()) for i in over_sampling_strategy.keys()]
    over_data_indices = [k for i in over_data_indices for k in i]
    X_over = X[over_data_indices]
    y_over = y[over_data_indices]

    sm = RandomOverSampler(sampling_strategy = over_sampling_strategy)
    X_over_resampled, y_over_resampled = sm.fit_resample(X_over, y_over)

    # Get X and y for under samplings and usr Random under sampling
    under_sampling_indices = np.array(np.where(combination==False)).flatten().tolist()
    under_sampling_strategy = dict([list(sampling_strategy.items())[i] for i in under_sampling_indices])
    under_data_indices = [list(np.array(np.where(data==[i])[0]).flatten()) for i in under_sampling_strategy.keys()]
    under_data_indices = [k for i in under_data_indices for k in i]
    X_under = X[under_data_indices]
    y_under = y[under_data_indices]

    sm = RandomUnderSampler(sampling_strategy = under_sampling_strategy)
    X_under_resampled, y_under_resampled = sm.fit_resample(X_under, y_under)
    X_resampled = np.array(X_over_resampled.flatten().tolist() + X_under_resampled.flatten().tolist()).reshape(-1, 1)
    y_resampled = np.array(y_over_resampled.flatten().tolist() + y_under_resampled.flatten().tolist())
  else:
    # use SMOTE oversampling
    sm = RandomOverSampler(sampling_strategy = sampling_strategy)
    X_resampled, y_resampled = sm.fit_resample(X, y)

  # get csv rows of resampled X
  annotations = list(map(src.get, X_resampled.flatten()))
  for annotation, label in zip(annotations, y_resampled):
    indices=np.array(np.where(data==str(annotation))).flatten().tolist()
    csv_rows.append(data[indices[0]].tolist())

  # genererate new train CSVs for each balancing rule
  with open("../data/source_S155/annotations/train_{}.csv".format(experiment['name']), 'w') as csv_file:
    writer = csv.writer(csv_file)
    header = ['annotation_label_id','label_id','label_name','label_hierarchy',
              'user_id','firstname','lastname','image_id','filename','image_longitude',
              'image_latitude','shape_id','shape_name','points','attributes']
    writer.writerow(header)
    for row in csv_rows:
        writer.writerow(row)
