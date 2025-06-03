from ultralytics import YOLO
import torch

def load_model(model_path="models/yolo11n.pt"):
  model = YOLO(model_path)
  model.to("cuda")
  return model
  
def detect_objects(model, frame):
  frame_tensor = torch.from_numpy(frame).to("cuda") if torch.cuda.is_available() else frame
  results = model(frame_tensor)
  return results
