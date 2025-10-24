# detect.py

from ultralytics import YOLO
import cv2 # Import OpenCV (diperlukan untuk tombol 'q')

# 1. Muat "otak" pintar yang sudah kita latih.
# Pastikan path ini benar
model = YOLO('C:/Users/ASUS/OneDrive/Desktop/proyek yolo/runs/detect/train/weights/best.pt') 

# 2. Tentukan sumber gambar/video.
# Ganti dengan path lengkap ke file video Anda.
# Contoh di Mac: '/Users/limwawin/Downloads/video_proyek.mp4'
# Contoh di Windows: 'C:/Users/limwawin/Downloads/video_proyek.mp4' 
source = 'C:/Users\ASUS\OneDrive\Desktop\proyek yolo\WhatsApp Video 2025-10-24 at 18.08.37.mp4' 

# 3. Jalankan deteksi dan tampilkan hasilnya secara real-time.
# show=True akan otomatis membuka jendela tampilan.
# stream=True lebih efisien untuk video.
# save=True (opsional) jika Anda ingin menyimpan hasil video deteksi.
results = model(source, show=True, stream=True, save=True) 

# Loop untuk menjaga jendela tetap terbuka dan mendeteksi penekanan tombol 'q'
for r in results:
    # Baris ini tidak wajib, hanya contoh akses data
    boxes = r.boxes 
    
    # Cek jika tombol 'q' ditekan untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup semua jendela OpenCV setelah loop berhenti
cv2.destroyAllWindows()
print("--- Deteksi pada video selesai ---")
# Jika save=True, hasil video disimpan di folder runs/detect/predict