import cv2
import time
import os
from ultralytics import YOLO
from openvino import Core

DRAW_KEYPOINTS = True
DRAW_SKELETON = True
DRAW_BOX = True
DRAW_LABEL = True
core = Core()
available_devices = core.available_devices
model = YOLO("yolo11n-pose_openvino_model/")
if 'GPU' in available_devices:
  if hasattr(model, 'model') and hasattr(model.model, 'ov_model'):
    try:
      compiled_model = core.compile_model(model.model.ov_model, 'GPU')
      model.model.ov_model = compiled_model
    except Exception as e:
      print(e)
else:
  os.environ["OV_CPU_THREADS_NUM"] = str(os.cpu_count())
  os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
  os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
cv2.setNumThreads(os.cpu_count())
cap = cv2.VideoCapture("0.mp4", cv2.CAP_FFMPEG)
if not cap.isOpened():
  exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 60)
skeleton = [
  (0, 1), (0, 2),
  (1, 3), (2, 4),
  (0, 5), (0, 6),
  (5, 7), (7, 9),
  (6, 8), (8, 10),
  (5, 11), (6, 12),
  (11, 13), (13, 15),
  (12, 14), (14, 16),
  (11, 12), (5, 6)
]
prev_time = time.time()
fps_count = 0.0
while True:
  ret, frame = cap.read()
  if not ret or frame is None:
    break
  sframe = cv2.resize(frame, (640, 360))
  results = model.predict(sframe, imgsz=640, verbose=False, max_det=20, half=False)[0]
  if results.keypoints and results.keypoints.xy is not None and results.keypoints.conf is not None:
    for kp, conf, box, score in zip(results.keypoints.xy, results.keypoints.conf, results.boxes.xyxy,
                                    results.boxes.conf):
      if score < 0.4:
        continue
      if DRAW_KEYPOINTS:
        for x, y in kp:
          cv2.circle(sframe, (int(x), int(y)), 2, (0, 255, 255), 1)
      if DRAW_SKELETON:
        for i, j in skeleton:
          if conf[i] > 0.5 and conf[j] > 0.5:
            x1, y1 = map(int, kp[i])
            x2, y2 = map(int, kp[j])
            cv2.line(sframe, (x1, y1), (x2, y2), (0, 128, 255), 1)
      x1, y1, x2, y2 = map(int, box)
      w, h = x2 - x1, y2 - y1
      aspect_ratio = w / h if h > 0 else 0
      falling = aspect_ratio > 1.2
      color = (0, 0, 255) if falling else (0, 255, 0)
      label = "Falling" if falling else "Normal"
      if DRAW_BOX:
        cv2.rectangle(sframe, (x1, y1), (x2, y2), color, 1)
      if DRAW_LABEL:
        cv2.putText(sframe, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_4)
  curr_time = time.time()
  elapsed = curr_time - prev_time
  fps_count = 1 / elapsed if elapsed > 0 else fps_count
  prev_time = curr_time
  cv2.putText(sframe, f"FPS: {fps_count:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
  cv2.imshow("VAK", sframe)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break
cap.release()
cv2.destroyAllWindows()
