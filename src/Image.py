import os
from pyvips import Image as VipsImage
import cv2
import numpy as np
import warnings
from IPython import embed
from keras.preprocessing.image import ImageDataGenerator
import ast



class Image(object):
   def __init__(self, path, distance, method):
      self.path = path
      self.filename = os.path.basename(path)
      self.distance = distance
      self.method = method
      self.annotations = []
      self.scale = 1

   def add_annotation(self, annotation):
      self.annotations.append(annotation)

   def set_target_distance(self, distance):
      self.scale = self.distance / distance
      for annotation in self.annotations:
         annotation.set_scale(self.scale)

   def generate_train_patches(self, images_path, masks_path, dimension, classes_dict):
      image = VipsImage.new_from_file(self.path).resize(self.scale)

      if image.width < dimension or image.height < dimension:
         warnings.warn('Image "{}" is smaller than the crop dimension!'.format(self.filename))


      classes_dict = {v: k for k, v in classes_dict.items()}
      classes = []

      for i, annotation in enumerate(self.annotations):
        if not annotation.label_id:
          # The ID of the interesting class is always 1.
          classes.append(1)
        else:
          # This is the ID of the class that is assigned in generate() above.
          classes.append(classes_dict[f"{annotation.label_id}"])

      masks = []

      for i, annotation in enumerate(self.annotations):
        mask = np.zeros((image.height, image.width), dtype=np.int32)
        cv2.circle(mask, annotation.get_center(), annotation.get_radius(), classes_dict[f"{annotation.label_id}"], -1)
        masks.append(mask)

      image_paths = []
      mask_paths = []
      mean_pixels = []

      for i, annotation in enumerate(self.annotations):
         image_file = '{}_{}.jpg'.format(self.filename, i)
         image_crop, mask_crops = self.generate_annotation_crop(image, masks, annotation, dimension)
         mask_file = self.save_mask(mask_crops, image_file, masks_path, classes)
         image_crop.write_to_file(os.path.join(images_path, image_file), strip=True, Q=95)
         image_paths.append(image_file)
         mask_paths.append(mask_file)
         np_crop = np.ndarray(buffer=image_crop.write_to_memory(), shape=[image_crop.height, image_crop.width, image_crop.bands], dtype=np.uint8)
         mean_pixels.append(np_crop.reshape((-1, 3)).mean(axis = 0))

      return image_paths, mask_paths, mean_pixels

   def generate_annotation_crop(self, image, masks, annotation, dimension):
      width, height = image.width, image.height

      crop_width = min(width, dimension)
      crop_height = min(height, dimension)
      current_crop_dimension = np.array([crop_width, crop_height])

      center = np.array(annotation.get_center())
      topLeft = np.round(center - current_crop_dimension / 2).astype(np.int32)
      bottomRight = np.round(center + current_crop_dimension / 2).astype(np.int32)
      offset = [0, 0]
      if topLeft[0] < 0: offset[0] = abs(topLeft[0])
      if topLeft[1] < 0: offset[1] = abs(topLeft[1])
      if bottomRight[0] > width: offset[0] = width - bottomRight[0]
      if bottomRight[1] > height: offset[1] = height - bottomRight[1]

      topLeft += offset
      bottomRight += offset

      image_crop = image.extract_area(topLeft[0], topLeft[1], current_crop_dimension[0], current_crop_dimension[1])
      mask_crops = [mask[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]] for mask in masks]

      if annotation.transformation and self.method == "tros":
        datagen = ImageDataGenerator()

        transformation = ast.literal_eval(annotation.transformation)
        img = np.fromstring(image_crop.write_to_memory(), dtype=np.uint8).reshape(image_crop.width, image_crop.height, 3)
        img_transformed = datagen.apply_transform(img, transformation)

        mask_transformation = {list(transformation.items())[0][0]: list(transformation.items())[0][1]}
        mask_crops = datagen.apply_transform(np.array(mask_crops), mask_transformation)

        img = img_transformed.reshape(image_crop.width * image_crop.height * 3)
        image_crop = VipsImage.new_from_memory(img.data, image_crop.width, image_crop.height, bands=3, format="uchar")

      return image_crop, mask_crops

   def save_mask(self, masks, filename, path, classes):
      combined = [(mask, class_id) for mask, class_id in zip(masks, classes) if np.any(mask)]
      mask_store = [mask for mask, class_id in combined]
      class_store = [class_id for mask, class_id in combined]
      mask_file = '{}.npz'.format(filename)
      np.savez_compressed(os.path.join(path, mask_file), masks=mask_store, classes=class_store)

      return mask_file

   def generate_random_crop(self, target_path, dimension, prefix=''):
      image = VipsImage.new_from_file(self.path)
      x = np.random.randint(image.width - dimension)
      y = np.random.randint(image.height - dimension)
      image_crop = image.extract_area(x, y, dimension, dimension)
      filename = self.filename
      if prefix != '':
         filename = '{}-{}'.format(prefix, filename)
      image_crop.write_to_file(os.path.join(target_path, filename), strip=True, Q=95)

      return filename

   def evaluate(self, results_path):
      total_annotations = len(self.annotations)
      results_file = os.path.join(results_path, '{}.png'.format(self.filename))
      results_image = VipsImage.new_from_file(results_file)
      results_image = np.ndarray(buffer=results_image.write_to_memory(), dtype=np.uint8, shape=[results_image.height, results_image.width, results_image.bands])

      contours, _ = cv2.findContours(results_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      detected_annotations = []
      correct_contours = set([])

      for annotation in self.annotations:
         detected = False
         for i, contour in enumerate(contours):
            distance = cv2.pointPolygonTest(contour, annotation.get_center(), True)
            # If the distance is negat1ive, the point is outside the polygon. Circles
            # that lie outside are still counted if they intersect the contour.
            if distance >= 0 or (annotation.get_radius() + distance) >= 0:
               detected = True
               correct_contours.add(i)
         if detected:
            detected_annotations.append(annotation)

      total_regions = len(contours)
      correct_detections = len(detected_annotations)
      correct_regions = len(correct_contours)

      return total_annotations, total_regions, correct_detections, correct_regions
