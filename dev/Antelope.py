import cv2
import time

cap = cv2.VideoCapture("0.mp4")
if not cap.isOpened():
  exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
fps = cap.get(cv2.CAP_PROP_FPS)
prevTime = time.time()
fpsCount = 0.0
while True:
  ret, frame = cap.read()
  if not ret or frame is None:
    break
  currTime = time.time()
  elapsedTime = currTime - prevTime
  if elapsedTime > 0:
    fpsCount = 1 / elapsedTime
    prevTime = currTime
  cv2.putText(frame, f'FPS: {fpsCount:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
  cv2.imshow('VAK', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()
