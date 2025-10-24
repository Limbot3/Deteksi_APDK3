# detect.py

from ultralytics import YOLO
import cv2

# 1. Load model yang sudah Anda latih
# Ganti dengan path ke file 'best.pt' Anda
model_path = 'C:/Users/ASUS/OneDrive/Desktop/proyek yolo/runs/detect/train/weights/best.pt' 
model = YOLO(model_path)

# 2. Pilih sumber untuk deteksi
# Ganti dengan path gambar atau video Anda, atau '0' untuk webcam.
source = 0 # Menggunakan webcam

# 3. Jalankan deteksi
# 'stream=True' lebih efisien untuk video atau webcam
results = model(source, stream=True, show=True) 

# Loop untuk menampilkan hasil secara real-time (hanya untuk video/webcam)
for r in results:
    # Baris ini tidak wajib, hanya untuk menunjukkan bahwa kita bisa mengakses data bounding box
    boxes = r.boxes 
    # cv2.waitKey(1) diperlukan agar jendela tampilan OpenCV tetap terbuka
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Deteksi dihentikan.")