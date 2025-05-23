def track_objects(frame, model):
  results = model.track(source=frame, tracker="bytetrack")
  return results