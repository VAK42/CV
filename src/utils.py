import cv2

def draw_bounding_boxes(frame, detections):
  for _, detection in detections.iterrows():
    x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']]
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
  return frame

def log_movement(movement_log, frame_number, detected_objects):
  with open(movement_log, "a") as log:
    log.write(f"{frame_number}:{len(detected_objects)}\n")