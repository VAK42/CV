import cv2
import pandas as pd
from src.config import VIDEO_SOURCE, MODEL_PATH, MOVEMENT_THRESHOLD
from src.detector import load_model, detect_objects
from src.processor import processor
from src.utils import draw_bounding_boxes, draw_skeleton, log_movement, classify_posture

def main():
  model = load_model(MODEL_PATH)
  cap = cv2.VideoCapture(VIDEO_SOURCE)
  prev_centers = []
  movement_log = "outputs/log"
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    results = detect_objects(model, frame)
    boxes = results[0].boxes
    keypoints = results[0].keypoints
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
      cv2.imshow("VAK", frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      continue
    df = pd.DataFrame(boxes.xyxy.cpu().numpy(), columns=['xmin', 'ymin', 'xmax', 'ymax'])
    df['conf'] = boxes.conf.cpu().numpy()
    df['cls'] = boxes.cls.cpu().numpy()
    centers = [
      ((row['xmin'] + row['xmax']) / 2, (row['ymin'] + row['ymax']) / 2)
      for _, row in df.iterrows()
    ]
    if prev_centers:
      movements = processor(prev_centers, centers, MOVEMENT_THRESHOLD)
      if any(movements):
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        log_movement(movement_log, frame_num, centers)
    postures = classify_posture(keypoints)
    frame = draw_bounding_boxes(frame, df, postures)
    frame = draw_skeleton(frame, keypoints)
    cv2.imshow("VAK", frame)
    prev_centers = centers
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()