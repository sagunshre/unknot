import numpy as np
from IPython import embed

class CircleAnnotation(object):
   def __init__(self, row):
      self.label_id = row[1]
      self.label = row[2]
      self.transformation = row[-1]
      self.annotation_id = row[0]
      print(self.annotation_id)
      center, radius = self.get_center_and_radius_(row)
      self.center = center
      self.radius = radius
      self.scale = 1

   def set_scale(self, factor):
      self.scale = factor

   def get_center(self):
      return tuple(np.round(self.center * self.scale).astype(int).tolist())

   def get_radius(self):
      return int(np.round(self.radius * self.scale))

   def get_center_and_radius_(self, row):
      points = np.array(row[13].strip('[]').split(','), dtype=float)
      shape_id = int(row[11])
      if shape_id == 1: # Point
         # Points get the default radius of 50
         return points[:2], 50
      elif shape_id == 4: # Circle
         return points[:2], points[2]
      elif points.size > 0:
         count = points.size;
         minX = np.PINF;
         minY = np.PINF;
         maxX = np.NINF;
         maxY = np.NINF;
         # points = points.reshape(-1, 2)
         for i in list(range(0, count-1, 2)):
           minX = min(minX, points[i])
           minY = min(minY, points[i+1])
           maxX = max(maxX, points[i])
           maxY = max(maxY, points[i+1])

         center = np.array([round((minX + maxX) / 2, 2), round((minY + maxY) / 2, 2)])

         radius = np.NINF
         for i in list(range(0, count-1, 2)):
            # radius = max(np.linalg.norm(center - points[i]), radius)
            radius = max(radius, (center[0] - points[i])**2 + (center[1] - points[i+1])**2)
         radius = round(np.sqrt(radius), 2)
         return center, radius
      else:
         raise Exception('Unsupported shape')

