# train.py

from ultralytics import YOLO

# 1. Load pre-trained model
# 'yolov8n.pt' adalah model kecil dan cepat, cocok untuk percobaan awal.
model = YOLO('yolov8n.pt')

# 2. Mulai training
# Gunakan raw string (r'...') agar backslash di path tidak error.
results = model.train(
    data=r"C:\Users\ASUS\OneDrive\Desktop\PPE_detection\data.yaml",  # path dataset
    epochs=50,     # jumlah epoch
    imgsz=640      # ukuran input gambar
)

print("✅ Training selesai!")
print("📁 Model dan hasil disimpan di folder 'runs/detect/train'")
