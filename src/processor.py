def processor(previous_centers, current_centers, threshold=20):
  movements = []
  for prev, curr in zip(previous_centers, current_centers):
    if abs(curr[0] - prev[0]) > threshold or abs(curr[1] - prev[1]) > threshold:
      movements.append(True)
    else:
      movements.append(False)
  return movements