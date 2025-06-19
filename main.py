import winsound
import math
import time
import csv
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from ultralytics import YOLO
from openvino import Core

videoPath = "0.mp4"
modelPath = "yolo11n-pose_openvino_model/"
expDir = "exp"
logFile = os.path.join(expDir, "exp.csv")
soundFile = "sound.wav"
frameWidth, frameHeight = 640, 360
maxDetections = 60

fallCooldown = 1.0
fallDisplayDuration = 1.0
minTrackingConfidence = 0.4
torsoAngleThreshold = 45
heightRatioThreshold = 0.3
hipsHeightThreshold = 0.85
velocityThreshold = 0.25

drawKeypoints = True
drawSkeleton = True
drawBox = True
drawLabel = True

preFallSec = 1.0
postFallSec = 1.0
fps = 60
bufferSize = int(preFallSec * fps)

os.makedirs(expDir, exist_ok=True)
if not os.path.exists(logFile):
  with open(logFile, "w", newline="") as f:
    csv.writer(f).writerow(["TIMESTAMP", "EVENT", "BOX_X1", "BOX_Y1", "BOX_X2", "BOX_Y2"])

core = Core()
cpuCount = os.cpu_count()
model = YOLO(modelPath)

if 'GPU' in core.available_devices and hasattr(model, 'model') and hasattr(model.model, 'ov_model'):
  try:
    compiled = core.compile_model(model.model.ov_model, 'GPU')
    model.model.ov_model = compiled
  except Exception as e:
    print("OpenVINO GPU Compile Failed: ", e)
else:
  os.environ.update({
    "OV_CPU_THREADS_NUM": str(cpuCount),
    "OMP_NUM_THREADS": str(cpuCount),
    "OPENBLAS_NUM_THREADS": str(cpuCount)
  })

cv2.setNumThreads(cpuCount)

cap = cv2.VideoCapture(videoPath, cv2.CAP_FFMPEG)
if not cap.isOpened():
  print("Could Not Open Video Source!")
  exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, fps)

skeletonPairs = [
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

executor = ThreadPoolExecutor(max_workers=4)
futures = []
trackedPersons = {}
nextPersonId = 1
lastFallTime = 0
fallDisplayTime = 0
lastCleanupTime = time.time()
lastFrame = None
lastResult = None
prevTime = time.time()
frameBuffer = deque(maxlen=bufferSize)
recording = False
recordStartTime = None
postFallFrames = 0
fallVideoWriter = None

def startfallvideo(tsx, width, height):
  fourcc = cv2.VideoWriter.fourcc(*'mp4v')
  filename = os.path.join(expDir, f"{tsx}.mp4")
  return cv2.VideoWriter(filename, fourcc, fps, (width, height))

while True:
  ret, frame = cap.read()
  if not ret:
    break

  resized = cv2.resize(frame, (frameWidth, frameHeight))
  frameBuffer.append(resized.copy())
  frameHeightNow = resized.shape[0]
  lastFrame = resized.copy()
  currentTime = time.time()

  if len(futures) < 1:
    futures.append(
      executor.submit(
        lambda img: model.predict(img, imgsz=640, verbose=False, max_det=maxDetections, half=False)[0],
        resized
      )
    )

  for f in futures[:]:
    if f.done():
      try:
        lastResult = f.result()
      except Exception as e:
        print("Inference Failed: ", e)
      futures.remove(f)

  if lastResult and lastResult.keypoints.xy is not None:
    for kp, conf, bbox, score in zip(lastResult.keypoints.xy, lastResult.keypoints.conf, lastResult.boxes.xyxy,
                                     lastResult.boxes.conf):
      if score < 0.4:
        continue

      if drawKeypoints:
        for x, y in kp:
          cv2.circle(lastFrame, (int(x), int(y)), 2, (0, 255, 255), 1)
      if drawSkeleton:
        for i, j in skeletonPairs:
          if conf[i] > 0.5 and conf[j] > 0.5:
            cv2.line(lastFrame, tuple(map(int, kp[i])), tuple(map(int, kp[j])), (0, 128, 255), 1)

      x1, y1, x2, y2 = map(int, bbox)
      falling = False
      requiredKp = [5, 6, 11, 12]

      if all(conf[i] > minTrackingConfidence for i in requiredKp):
        lSh, rSh = kp[5], kp[6]
        lHp, rHp = kp[11], kp[12]
        midSh = ((lSh[0] + rSh[0]) / 2, (lSh[1] + rSh[1]) / 2)
        midHp = ((lHp[0] + rHp[0]) / 2, (lHp[1] + rHp[1]) / 2)

        dx, dy = midHp[0] - midSh[0], midHp[1] - midSh[1]
        torsoLength = math.hypot(dx, dy)

        verticalAngle, heightRatio, hipsRatio, velocityIndicator = 90, 1.0, 0.0, False

        if torsoLength > 10:
          verticalAngle = math.degrees(math.acos(abs(dy) / torsoLength))
          heightRatio = abs(midSh[1] - midHp[1]) / torsoLength
          hipsRatio = midHp[1] / frameHeightNow

          avgPos = ((midSh[0] + midHp[0]) / 2, (midSh[1] + midHp[1]) / 2)
          bestId, minDist = None, float('inf')

          for pid, person in trackedPersons.items():
            prevAvg = ((person[0] + person[2]) / 2, (person[1] + person[3]) / 2)
            dist = math.hypot(avgPos[0] - prevAvg[0], avgPos[1] - prevAvg[1]) / frameHeightNow
            if dist < 0.1 and dist < minDist:
              bestId, minDist = pid, dist

          if bestId is not None:
            person = trackedPersons[bestId]
            tDiff = currentTime - person[4]
            if tDiff > 0:
              dySh = (midSh[1] - person[1]) / frameHeightNow
              dyHp = (midHp[1] - person[3]) / frameHeightNow
              velocity = ((dySh + dyHp) / 2) / tDiff
              velocityIndicator = velocity > velocityThreshold
            trackedPersons[bestId] = (*midSh, *midHp, currentTime)
          else:
            trackedPersons[nextPersonId] = (*midSh, *midHp, currentTime)
            nextPersonId += 1

        conds = [
          verticalAngle > torsoAngleThreshold,
          heightRatio < heightRatioThreshold,
          hipsRatio > hipsHeightThreshold
        ]
        falling = (
          (conds[0] and conds[1]) or
          (conds[0] and conds[2]) or
          (conds[1] and conds[2]) or
          (velocityIndicator and any(conds))
        )
      else:
        boxW, boxH = x2 - x1, y2 - y1
        falling = (boxW / boxH > 1.2) if boxH > 0 else False

      if falling and (currentTime - lastFallTime >= fallCooldown):
        lastFallTime = currentTime
        fallDisplayTime = currentTime
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(expDir, f"{ts}.jpg"), lastFrame)
        with open(logFile, "a", newline="") as f:
          csv.writer(f).writerow([ts, "Fall", x1, y1, x2, y2])
        winsound.PlaySound(soundFile, winsound.SND_FILENAME | winsound.SND_ASYNC)
        fallVideoWriter = startfallvideo(ts, frameWidth, frameHeight)
        for bf in frameBuffer:
          fallVideoWriter.write(bf)
        recording = True
        postFallFrames = int(postFallSec * fps)

      color = (0, 0, 255) if falling else (0, 255, 0)
      if DRAW_BOX:
        cv2.rectangle(lastFrame, (x1, y1), (x2, y2), color, 1)
      if DRAW_LABEL:
        label = "Falling" if falling else "Normal"
        cv2.putText(lastFrame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

  if currentTime - lastCleanupTime > 5:
    trackedPersons = {
      pid: person for pid, person in trackedPersons.items()
      if currentTime - person[4] < 2.0
    }
    lastCleanupTime = currentTime

  elapsed = currentTime - prevTime
  currentFps = 1.0 / elapsed if elapsed > 0 else 0.0
  prevTime = currentTime
  h, w = lastFrame.shape[:2]
  ts = time.strftime("%Y-%m-%d %H:%M:%S")
  fallDisplay = 1 if currentTime - fallDisplayTime < fallDisplayDuration else 0
  infoText = f"FPS: {currentFps:.2f}  |  Fall: {fallDisplay}  |  Res: {w}x{h}  |  Time: {ts}"
  cv2.putText(lastFrame, infoText, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

  cooldownElapsed = currentTime - lastFallTime
  if 0 <= cooldownElapsed < fallCooldown:
    ratio = min(cooldownElapsed / fallCooldown, 1.0)
    angle = int(360 * ratio)
    center = (w - 15, h - 15)
    radius = 6
    cv2.circle(lastFrame, center, radius, (60, 60, 60), -1)
    cv2.ellipse(lastFrame, center, (radius, radius), -90, 0, angle, (0, 255, 255), -1)

  if recording and fallVideoWriter:
    fallVideoWriter.write(lastFrame)
    postFallFrames -= 1
    if postFallFrames <= 0:
      fallVideoWriter.release()
      recording = False
      fallVideoWriter = None

  cv2.imshow("VAK", lastFrame)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()
executor.shutdown()
if fallVideoWriter:
  fallVideoWriter.release()
