from ultralytics import YOLO

def load_model(model_path="models/yolo11n.pt"):
  model = YOLO(model_path)
  model.to("cuda")
  return model
  
def detect_objects(model, frame):
  zframe = cv2.resize(frame, (640, 640))
  results = model(zframe)
  return results
