from ultralytics import YOLO
import torch

model = YOLO('yolov8x')
print(torch.cuda.is_available())
results = model.predict('input_data/08fd33_4.mp4', save=True, device='cuda')    #device=0

print(results[0])
print('=========================================')

for box in results[0].boxes:
    print(box)
