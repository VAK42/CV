import cv2
import time
from ultralytics import YOLO
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("yolo11n-pose.pt")
cap = cv2.VideoCapture("0.mp4")
if not cap.isOpened():
  exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
skeleton = [
  (0, 1), (0, 2),
  (1, 3), (2, 4),
  (0, 5), (0, 6),
  (5, 7), (7, 9),
  (6, 8), (8, 10),
  (5, 11), (6, 12),
  (11, 13), (13, 15),
  (12, 14), (14, 16),
  (11, 12),
  (5, 6)
]
prev_time = time.time()
fps_count = 0.0
while True:
  ret, frame = cap.read()
  if not ret or frame is None:
    break
  results = model.predict(frame, verbose=False, device="cpu")[0]
  if results.keypoints and results.keypoints.xy is not None and results.keypoints.conf is not None:
    for kp, conf in zip(results.keypoints.xy, results.keypoints.conf):
      for x, y in kp:
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
      for i, j in skeleton:
        if conf[i] > 0.5 and conf[j] > 0.5:
          x1, y1 = map(int, kp[i])
          x2, y2 = map(int, kp[j])
          cv2.line(frame, (x1, y1), (x2, y2), (0, 128, 255), 2)
  curr_time = time.time()
  elapsed = curr_time - prev_time
  fps_count = 1 / elapsed if elapsed > 0 else fps_count
  prev_time = curr_time
  cv2.putText(frame, f"FPS: {fps_count:.2f}", (10, 50),
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  cv2.imshow("VAK", frame)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break
cap.release()
cv2.destroyAllWindows()