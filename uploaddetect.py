# detect.py

from ultralytics import YOLO
import cv2 

model = YOLO('C:/Users/ASUS/OneDrive/Desktop/proyek yolo/runs/detect/train/weights/best.pt') 

source = 'C:/Users\ASUS\OneDrive\Desktop\proyek yolo\WhatsApp Video 2025-10-24 at 18.08.37.mp4' 

results = model(source, show=True, stream=True, save=True) 

for r in results:
    boxes = r.boxes 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("--- Deteksi pada video selesai ---")
