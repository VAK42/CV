import cv2
import time

cap = cv2.VideoCapture("0.mp4")
if not cap.isOpened():
  exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
fps = cap.get(cv2.CAP_PROP_FPS)
prev_time = time.time()
fps_count = 0.0
while True:
  ret, frame = cap.read()
  if not ret or frame is None:
    break
  curr_time = time.time()
  elapsed_time = curr_time - prev_time
  if elapsed_time > 0:
    fps_count = 1 / elapsed_time
    prev_time = curr_time
  cv2.putText(frame, f'FPS: {fps_count:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
  cv2.imshow('VAK', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()