import winsound
import time
import cv2
import csv
import os
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from openvino import Core

EXP_DIR = "exp"
LOG_FILE = os.path.join(EXP_DIR, "exp.csv")
LAST_FALL_TIME = 0
FALL_DISPLAY_TIME = 0
FALL_COOLDOWN = 1.0
FALL_DISPLAY_DURATION = 1.0
DRAW_KEYPOINTS = True
DRAW_SKELETON = True
DRAW_BOX = True
DRAW_LABEL = True
os.makedirs(EXP_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
  with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["TIMESTAMP", "EVENT", "BOX_X1", "BOX_Y1", "BOX_X2", "BOX_Y2"])
core = Core()
availableDevices = core.available_devices
model = YOLO("yolo11n-pose_openvino_model/")
if 'GPU' in availableDevices:
  if hasattr(model, 'model') and hasattr(model.model, 'ov_model'):
    try:
      compiledModel = core.compile_model(model.model.ov_model, 'GPU')
      model.model.ov_model = compiledModel
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
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 60)
executor = ThreadPoolExecutor(max_workers=4)
futures = []
lastResults = None
lastFrameForDraw = None
prevTime = time.time()
fpsCount = 0.0
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
while True:
  ret, frame = cap.read()
  if not ret or frame is None:
    break
  sframe = cv2.resize(frame, (640, 360))
  lastFrameForDraw = sframe.copy()
  if len(futures) < 1:
    futures.append(
      executor.submit(lambda img: model.predict(img, imgsz=640, verbose=False, max_det=60, half=False)[0], sframe))
  for f in futures[:]:
    if f.done():
      try:
        lastResults = f.result()
      except Exception as e:
        print(e)
      futures.remove(f)
  if lastResults and lastResults.keypoints and lastResults.keypoints.xy is not None and lastResults.keypoints.conf is not None:
    for kp, conf, box, score in zip(lastResults.keypoints.xy, lastResults.keypoints.conf, lastResults.boxes.xyxy,
                                    lastResults.boxes.conf):
      if score < 0.4:
        continue
      if DRAW_KEYPOINTS:
        for x, y in kp:
          cv2.circle(lastFrameForDraw, (int(x), int(y)), 2, (0, 255, 255), 1)
      if DRAW_SKELETON:
        for i, j in skeleton:
          if conf[i] > 0.5 and conf[j] > 0.5:
            x1, y1 = map(int, kp[i])
            x2, y2 = map(int, kp[j])
            cv2.line(lastFrameForDraw, (x1, y1), (x2, y2), (0, 128, 255), 1)
      x1, y1, x2, y2 = map(int, box)
      w, h = x2 - x1, y2 - y1
      aspectRatio = w / h if h > 0 else 0
      falling = aspectRatio > 1.2
      if falling:
        currentTime = time.time()
        if currentTime - LAST_FALL_TIME >= FALL_COOLDOWN:
          LAST_FALL_TIME = currentTime
          FALL_DISPLAY_TIME = currentTime
          timestamp = time.strftime("%Y%m%d_%H%M%S")
          cv2.imwrite(os.path.join(EXP_DIR, f"{timestamp}.jpg"), lastFrameForDraw)
          with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, "Fall", x1, y1, x2, y2])
          winsound.PlaySound("sound.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
      color = (0, 0, 255) if falling else (0, 255, 0)
      label = "Falling" if falling else "Normal"
      if DRAW_BOX:
        cv2.rectangle(lastFrameForDraw, (x1, y1), (x2, y2), color, 1)
      if DRAW_LABEL:
        cv2.putText(lastFrameForDraw, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_4)
  currTime = time.time()
  elapsed = currTime - prevTime
  fpsCount = 1 / elapsed if elapsed > 0 else fpsCount
  prevTime = currTime
  if lastFrameForDraw is not None:
    height, width = lastFrameForDraw.shape[:2]
    timestampStr = time.strftime("%Y-%m-%d %H:%M:%S")
    fallCountDisplay = 1 if time.time() - FALL_DISPLAY_TIME < FALL_DISPLAY_DURATION else 0
    infoText = f"FPS: {fpsCount:.2f}  |  Fall: {fallCountDisplay}  |  Res: {width}x{height}  |  Time: {timestampStr}"
    cv2.putText(lastFrameForDraw, infoText, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cooldownElapsed = time.time() - LAST_FALL_TIME
    if 0 <= cooldownElapsed < FALL_COOLDOWN:
      cooldownRatio = min(cooldownElapsed / FALL_COOLDOWN, 1.0)
      center = (width - 30, height - 30)
      radius = 4
      thickness = 2
      cv2.circle(lastFrameForDraw, center, radius, (50, 50, 50), thickness)
      angle = int(360 * cooldownRatio)
      cv2.ellipse(lastFrameForDraw, center, (radius, radius), -90, 0, angle, (0, 255, 255), thickness)
    cv2.imshow("VAK", lastFrameForDraw)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break
cap.release()
cv2.destroyAllWindows()
executor.shutdown()