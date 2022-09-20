import sys
import os
import csv
import yaml
import json
import numpy as np
from os.path import exists
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from IPython import embed

class Sampling(object):
  def __init__(self, report_path, config):
    self.path = report_path
    self.config = self.get_config(config)
    self.data = np.array(self.read_report(), dtype="U256")
    self.total_patches = self.data.shape[0]
    self.sorted_labels = self.get_sorted_label_strategy()
    self.src = dict(list(zip(list(range(0,len(self.data[:, 0].tolist()))), self.data[:, 0].tolist())))
    self.X = np.array(list(self.src.keys())).reshape(-1, 1)
    self.y = self.data[:, 1]

  def read_report(self):
    data = []
    with open(self.path, 'r') as file:
      reader = csv.reader(file)
      next(reader)
      for row in reader:
        data.append(row)
    return data

  def get_config(self, config):
    with open(config, 'r') as file:
      experiments = yaml.unsafe_load(file)
    return experiments

  def get_sorted_label_strategy(self):
    counter =  Counter(self.data[:, 1])
    return Counter(dict(sorted(counter.items(), key=lambda pair: pair[1], reverse=True)))


  # Balancing rule
  def sample(self):
    for key, rule in enumerate(self.config["rules"]):
      name = rule['name']
      sample_sizes = dict(zip(list(self.sorted_labels.keys()), [0]*len(self.sorted_labels.keys())))

      for k, v in self.sorted_labels.items():
        label_maj = list(self.sorted_labels.values())[0]

        if(v == label_maj):
          sample_sizes[k] = int(round(label_maj))
          continue

        if(name == "rule-50"):
          if(v < label_maj/2):
            sample_sizes[k] = int(round(label_maj/2))
          else:
            sample_sizes[k] = int(round(self.sorted_labels[k]))

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

      self.config["rules"][key]['sample_sizes'] = sample_sizes
    return self.config


  # SMOTE and random undersampling
  def generateRandomSamples(self, experiments):
    for experiment in experiments["rules"]:
      sampling_strategy = experiment["sample_sizes"]
      # check if any label needs to be under sampled
      combination = np.greater_equal(np.array(list(sampling_strategy.values())), np.array(list(self.sorted_labels.values())))
      if(np.any(combination==False)):
        # for oversampling and under sampling rule 50-100

        # Get X and y for over samplings and use SMOTE
        over_sampling_indices = np.array(np.where(combination==True)).flatten().tolist()
        over_sampling_strategy = dict([list(sampling_strategy.items())[i] for i in over_sampling_indices])
        over_data_indices = [list(np.array(np.where(self.data==[i])[0]).flatten()) for i in over_sampling_strategy.keys()]
        over_data_indices = [k for i in over_data_indices for k in i]
        X_over = self.X[over_data_indices]
        y_over = self.y[over_data_indices]

        sm = RandomOverSampler(sampling_strategy = over_sampling_strategy)
        X_over_resampled, y_over_resampled = sm.fit_resample(X_over, y_over)

        # Get X and y for under samplings and usr Random under sampling
        under_sampling_indices = np.array(np.where(combination==False)).flatten().tolist()
        under_sampling_strategy = dict([list(sampling_strategy.items())[i] for i in under_sampling_indices])
        under_data_indices = [list(np.array(np.where(self.data==[i])[0]).flatten()) for i in under_sampling_strategy.keys()]
        under_data_indices = [k for i in under_data_indices for k in i]
        X_under = self.X[under_data_indices]
        y_under = self.y[under_data_indices]

        sm = RandomUnderSampler(sampling_strategy = under_sampling_strategy)
        X_under_resampled, y_under_resampled = sm.fit_resample(X_under, y_under)
        X_resampled = np.array(X_over_resampled.flatten().tolist() + X_under_resampled.flatten().tolist()).reshape(-1, 1)
        y_resampled = np.array(y_over_resampled.flatten().tolist() + y_under_resampled.flatten().tolist())
      else:
        # use SMOTE oversampling
        sm = RandomOverSampler(sampling_strategy = sampling_strategy)
        X_resampled, y_resampled = sm.fit_resample(self.X, self.y)

      self.saveSampleCSV(experiment['name'], X_resampled, y_resampled)

  def generateTranformationOverSamples(self):
    return True

  def saveSampleCSV(self, exp_name, X, y):
    csv_rows = []

    # get csv rows of resampled X
    annotations = list(map(self.src.get, X.flatten()))
    for annotation, label in zip(annotations, y):
      indices=np.array(np.where(self.data==str(annotation))).flatten().tolist()
      csv_rows.append(self.data[indices[0]].tolist())

    # genererate new train CSVs for each balancing rule
    with open("../data/source_S155/annotations/train_{}.csv".format(exp_name), 'w') as csv_file:
      writer = csv.writer(csv_file)
      header = ['annotation_label_id','label_id','label_name','label_hierarchy',
                'user_id','firstname','lastname','image_id','filename','image_longitude',
                'image_latitude','shape_id','shape_name','points','attributes']
      writer.writerow(header)
      for row in csv_rows:
          writer.writerow(row)


exp_config_path = "exp_config.yml"
file_exists = exists(exp_config_path)
if file_exists:
  config = {'methods': ['smote', 'tros'],
            'rules': [{'name': 'rule-50', 'sample_sizes': {}},
                      {'name': 'rule-75', 'sample_sizes': {}},
                      {'name': 'rule-50-75-100', 'sample_sizes': {}},
                      {'name': 'rule-50-100', 'sample_sizes': {}}]}

  with open(exp_config_path, 'w') as exp_config:
    yaml.dump(config, exp_config)

sampling = Sampling("../data/source_S155/annotations/train.csv", exp_config_path)
config = sampling.sample()
with open(exp_config_path, 'w') as exp_config:
  yaml.dump(json.loads(json.dumps(config)), exp_config)

for method in config["methods"]:
  if method == "smote":
    sampling.generateRandomSamples(config)
  else:
    sampling.generateTranformationOverSamples()