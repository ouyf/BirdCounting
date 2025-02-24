from ultralytics import YOLO
import torch


model = YOLO('.\\best.pt')


model.eval()


results = model('.\\test10.jpg') 

num_objects = len(results[0].boxes)

print(f"Detected {num_objects} objects.")

results[0].show()