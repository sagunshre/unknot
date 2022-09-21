import sys
import os
import csv
import yaml
import json
import numpy as np
from os.path import exists
from collections import Counter
from itertools import islice, cycle, groupby
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from IPython import embed

class Sampling(object):
  def __init__(self, report_path, config):
    self.report_path = report_path
    self.config_name = "exp_config.yml"
    self.config = config
    self.csv_rows = []

  def configure(self):
    self.data = np.array(self.read_report(), dtype="U256")
    self.total_patches = self.data.shape[0]
    self.sorted_labels = self.get_sorted_label_strategy()

  def start(self):
    self.configure()
    config = self.sample()
    with open(self.config_name, 'w') as exp_config:
      yaml.dump(json.loads(json.dumps(config)), exp_config)
    # for method in config["methods"]:
    getattr(self, "randomSampling")("randomSampling")

  def read_report(self):
    data = []
    with open(self.report_path, 'r') as file:
      reader = csv.reader(file)
      next(reader)
      for row in reader:
        data.append(row)
    return data

  def get_sorted_label_strategy(self):
    counter =  Counter(self.data[:, 1])
    return Counter(dict(sorted(counter.items(), key=lambda pair: pair[1], reverse=True)))


  # Balancing rule
  def sample(self):
    for key, rule in enumerate(self.config["rules"]):
      name = rule['name']
      sample_sizes = dict.fromkeys(self.sorted_labels.keys(),0)

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
  def randomSampling(self, method):
    self.src = self.data[:, 0]
    self.X = np.array(range(0, self.src.size)).reshape(-1, 1)
    self.y = self.data[:, 1]

    for experiment in self.config["rules"]:
      sampling_strategy = experiment["sample_sizes"]
      # check if any label needs to be under sampled
      combination = np.greater_equal(list(sampling_strategy.values()), list(self.sorted_labels.values()))
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

      self.saveSampleCSV(method, experiment, X_resampled, y_resampled)

  def saveSampleCSV(self, method, exp, X, y):
    # get csv rows of resampled X
    exp_name = exp['name']
    annotations = list(self.src[X.flatten()])
    annotations = list(zip(annotations, y))
    annotations.sort(key = lambda x: x[0])

    label_maj = list(self.sorted_labels.keys())[0]
    majority_samples = [i for i in annotations if i[1] == label_maj]
    minority_samples = [i for i in annotations if i[1] != label_maj]

    # Grouping identical annotation ids
    keyfunc = lambda a: a[0]
    grouper = groupby(minority_samples, keyfunc)
    groups = [list(minority_samples) for _, minority_samples in grouper]

    annotation_labels = list(map(list, majority_samples))
    list(map(lambda x: x.append(""), annotation_labels))

    for group in groups:
      transformation = list(islice(cycle(self.config['transformation']), len(group)))
      group = list(map(list, group))
      list(map(lambda el,i: el.append(transformation[i]), group, range(0,len(group))))
      annotation_labels = annotation_labels + group

    for annotation, label, transformation in annotation_labels:
      indices=np.array(np.where(self.data==str(annotation))).flatten().tolist()
      row = self.data[indices[0]].tolist()
      row.append('{}'.format(transformation))
      self.csv_rows.append(row)

    # genererate new train CSVs for each balancing rule
    path = "../data/source_S155/annotations/{}".format("sampling")
    if not os.path.exists(path):
      os.makedirs(path)

    with open("{}/train_{}.csv".format(path, exp_name), 'w') as csv_file:
      writer = csv.writer(csv_file)
      header = ['annotation_label_id','label_id','label_name','label_hierarchy',
                'user_id','firstname','lastname','image_id','filename','image_longitude',
                'image_latitude','shape_id','shape_name','points','attributes', 'transformation']
      writer.writerow(header)
      for row in self.csv_rows:
          writer.writerow(row)


config = { 'methods': ['randomSampling', 'transformationSampling'],
                    'rules': [{'name': 'rule-50', 'sample_sizes': {}},
                              {'name': 'rule-75', 'sample_sizes': {}},
                              {'name': 'rule-50-75-100', 'sample_sizes': {}},
                              {'name': 'rule-50-100', 'sample_sizes': {}}],
                    'transformation': [{"theta": 15, "channel_shift_intensity": 30},
                                       {"theta": 15, "channel_shift_intensity": -30},
                                       {"theta": -15, "channel_shift_intensity": 30},
                                       {"theta": -15, "channel_shift_intensity": -30},
                                       {"flip_horizontal": True , "channel_shift_intensity": 30},
                                       {"flip_horizontal": True, "channel_shift_intensity": -30},
                                       {"flip_vertical": True, "channel_shift_intensity": 30},
                                       {"flip_vertical": True, "channel_shift_intensity": -30}]}

sampling = Sampling(sys.argv[1], config)
sampling.start()
