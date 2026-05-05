import torch
import tensorflow as tf
import sys

print("Python Versiyonu:", sys.version)
print("\n--- PyTorch Kontrolü ---")
cuda_available = torch.cuda.is_available()
print(f"CUDA Mevcut mu?: {cuda_available}")
if cuda_available:
    print(f"Ekran Kartı İsmi: {torch.cuda.get_device_name(0)}")
else:
    print("UYARI: PyTorch ekran kartını görmüyor. (CPU modunda çalışacak)")

print("\n--- TensorFlow Kontrolü ---")
try:
    gpu_list = tf.config.list_physical_devices('GPU')
    print(f"GPU Sayısı: {len(gpu_list)}")
    if gpu_list:
        print(f"Detaylar: {gpu_list}")
    else:
        print("UYARI: TensorFlow ekran kartını görmüyor.")
except Exception as e:
    print(f"Hata: {e}")
