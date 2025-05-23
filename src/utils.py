import cv2

def draw_bounding_boxes(frame, detections, labels=None):
  for i, (_, detection) in enumerate(detections.iterrows()):
    x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    if labels and i < len(labels):
      cv2.putText(frame, labels[i], (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
  return frame

def log_movement(movement_log, frame_number, detected_objects):
  with open(movement_log, "a") as log:
    log.write(f"{frame_number}:{len(detected_objects)}\n")

def classify_posture(keypoints):
  labels = []
  for kp in keypoints:
    coords = kp.xy[0]
    confs = kp.conf[0]

    def avg(pt1, pt2):
      return ((coords[pt1][0] + coords[pt2][0]) / 2,
              (coords[pt1][1] + coords[pt2][1]) / 2)

    if all(confs[i] > 0.5 for i in [5, 6, 11, 12, 15, 16]):
      shoulder_y = avg(5, 6)[1]
      ankle_y = avg(15, 16)[1]
      height = ankle_y - shoulder_y
      if height < 100:
        labels.append("Falling")
      else:
        labels.append("Normal")
    else:
      labels.append("Normal")
  return labels

LIMBS = [
  (5, 7), (7, 9),
  (6, 8), (8, 10),
  (5, 6),
  (11, 13), (13, 15),
  (12, 14), (14, 16),
  (11, 12),
  (5, 11), (6, 12)
]

def draw_skeleton(frame, keypoints):
  for kp in keypoints:
    coords = kp.xy[0]
    confs = kp.conf[0]
    for (x, y), c in zip(coords, confs):
      if c > 0.5:
        cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
    for i, j in LIMBS:
      if confs[i] > 0.5 and confs[j] > 0.5:
        pt1 = tuple(map(int, coords[i]))
        pt2 = tuple(map(int, coords[j]))
        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
  return frame