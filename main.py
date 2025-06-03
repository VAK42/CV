import pandas as pd
import logging
import torch
import time
import cv2
import os
from src.config import VIDEO_SOURCE, MODEL_PATH, MOVEMENT_THRESHOLD
from src.detector import load_model, detect_objects
from src.processor import processor
from src.utils import draw_bounding_boxes, draw_skeleton, log_movement, classify_posture

os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
  filename='outputs/cv.log',
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
  print(torch.cuda.is_available())
  logging.info("Application Started!")
  cap = None
  out = None
  try:
    model = load_model(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
      logging.error("Failed To Open Video Source!")
      return
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    fps_original = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('outputs/Output.mp4', fourcc, fps_original, (width, height))
    prev_centers = []
    prev_time = time.time()
    movement_log_path = "outputs/vak"
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        logging.warning("Failed To Read Frame!")
        break
      try:
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
        dynamic_threshold = MOVEMENT_THRESHOLD
        if len(df) > 0:
          avg_height = (df['ymax'] - df['ymin']).mean()
          dynamic_threshold = max(10, min(50, avg_height * 0.25))
        if prev_centers:
          movements = processor(prev_centers, centers, dynamic_threshold)
          if any(movements):
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            log_movement(movement_log_path, frame_num, centers)
            logging.info(f"Movement Logged At Frame {frame_num} With {len(centers)} Center(s)")
        postures = classify_posture(keypoints)
        frame = draw_bounding_boxes(frame, df, postures)
        frame = draw_skeleton(frame, keypoints)
        if "Falling" in postures:
          frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
          snapshot_path = f"outputs/{frame_num}.jpg"
          cv2.imwrite(snapshot_path, frame)
          logging.info(f"Snapshot Saved: {snapshot_path}")
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        out.write(frame)
        cv2.imshow("VAK", frame)
        prev_centers = centers
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
      except Exception as e:
        logging.exception("Error During Frame Processing: %s", e)
  except Exception as e:
    logging.exception("Unexpected Error Occurred In Main(): %s", e)
  finally:
    logging.info("Application Exiting")
    if cap is not None:
      cap.release()
    if out is not None:
      out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
