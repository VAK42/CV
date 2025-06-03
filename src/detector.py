from ultralytics import YOLO

def load_model(model_path="models/yolo11n.pt"):
  model = YOLO(model_path)
  model.to("cuda")
  return model
  
def detect_objects(model, frame):
  results = model(frame)
  return results
