# Hasil eksperimen historis

Seluruh angka berasal dari 16 eksperimen full yang sudah selesai pada proyek lokal. Test set berisi 624 citra (234 NORMAL, 390 PNEUMONIA); threshold 0,5. Angka ditampilkan sebagai hasil historis, tanpa menyertakan dataset, prediksi per citra, atau checkpoint.

Nilai lengkap, termasuk throughput dan latency Tensor-E2E, tersedia pada [experiment_results.csv](experiment_results.csv).

## v1: PneuNet dan empat pembanding, seluruhnya dari awal

### Baseline

| Model | Accuracy | F1 macro | Parameter | GPU mean (ms) | GPU P95 (ms) |
|---|---:|---:|---:|---:|---:|
| PneuNet | 76.44% | 0.6930 | 142,412 | 5.57 | 8.26 |
| ResNet50_Attention | 76.76% | 0.6996 | 24,036,545 | 9.77 | 13.71 |
| DenseNet121 | 80.61% | 0.7594 | 6,954,881 | 25.12 | 32.91 |
| EfficientNetB4 | 90.38% | 0.8928 | 17,550,409 | 30.38 | 39.24 |
| VGG16 | 62.50% | 0.3846 | 14,715,201 | 5.11 | 5.44 |

### CLAHE + augmentasi

| Model | Accuracy | F1 macro | Parameter | GPU mean (ms) | GPU P95 (ms) |
|---|---:|---:|---:|---:|---:|
| PneuNet | 82.37% | 0.7867 | 142,412 | 5.30 | 7.39 |
| ResNet50_Attention | 78.21% | 0.7271 | 24,036,545 | 10.49 | 14.55 |
| DenseNet121 | 85.42% | 0.8291 | 6,954,881 | 25.79 | 37.54 |
| EfficientNetB4 | 84.62% | 0.8189 | 17,550,409 | 28.57 | 38.38 |
| VGG16 | 62.50% | 0.3846 | 14,715,201 | 5.16 | 5.27 |

## v2: PneuNet revisi dan ResNet50 CBAM, keduanya dari awal

### Baseline

| Model | Accuracy | F1 macro | Parameter | GPU mean (ms) | GPU P95 (ms) |
|---|---:|---:|---:|---:|---:|
| PneuNet | 82.37% | 0.7860 | 157,224 | 7.26 | 11.40 |
| ResNet50_CBAM | 77.56% | 0.7138 | 24,034,467 | 13.59 | 18.96 |

### CLAHE + augmentasi

| Model | Accuracy | F1 macro | Parameter | GPU mean (ms) | GPU P95 (ms) |
|---|---:|---:|---:|---:|---:|
| PneuNet | 82.53% | 0.7883 | 157,224 | 7.14 | 10.12 |
| ResNet50_CBAM | 84.46% | 0.8204 | 24,034,467 | 13.45 | 17.59 |

## v3: ResNet50 pretrained ImageNet-1K V2, fine-tuning seluruh layer

### Baseline

| Model | Accuracy | F1 macro | Parameter | GPU mean (ms) | GPU P95 (ms) |
|---|---:|---:|---:|---:|---:|
| ResNet50_Pretrained | 80.13% | 0.7555 | 23,510,081 | 11.42 | 16.90 |

### CLAHE + augmentasi

| Model | Accuracy | F1 macro | Parameter | GPU mean (ms) | GPU P95 (ms) |
|---|---:|---:|---:|---:|---:|
| ResNet50_Pretrained | 83.65% | 0.8093 | 23,510,081 | 11.54 | 17.18 |

## Cara membaca latency

- Perangkat: NVIDIA GeForce RTX 3060 Laptop GPU, PyTorch CUDA FP32, input 224 x 224, batch 1, 20 warm-up, 624 citra test.
- **GPU mean/P95:** rata-rata/persentil ke-95 waktu forward yang diukur memakai CUDA Events. Transfer CPU ke GPU tidak termasuk. P95 bukan interval kepercayaan.
- **Tensor-E2E:** dimulai dari tensor CPU siap pakai, transfer ke GPU, forward, sigmoid, sampai hasil kembali ke CPU. Pembacaan/dekode gambar, letterbox, CLAHE, serta pembuatan augmentasi di luar timer.
- **Throughput:** citra/detik dari pengukuran Tensor-E2E berurutan pada batch 1; bukan throughput maksimum pada batch besar.
- Baseline dan CLAHE+aug memakai arsitektur yang sama. Selisih latency keduanya tidak mengukur biaya CLAHE karena preprocessing berada di luar timer.
- Pengukuran v1, v2, v3 dilakukan pada run berbeda. Perbandingan antar-run bersifat deskriptif; benchmark bersama belum dilakukan.
- Paper melaporkan Raspberry Pi 4, RAM 4 GB, CPU-only, TensorFlow Lite. Hasil GPU ini tidak membuktikan percepatan terhadap angka 20 ms pada paper. FLOPs dan latency perangkat edge implementasi ini belum diukur.

## Interpretasi

PneuNet v1 CLAHE+aug mencapai 82,37% / F1 macro 0,7867 pada 5,30 ms. DenseNet121 CLAHE+aug mencapai 85,42% / 0,8291 pada 25,79 ms. PneuNet memiliki latency sekitar 4,87 kali lebih cepat dengan selisih accuracy 3,04 poin persentase.

PneuNet v2 CLAHE+aug mencapai 82,53% / 0,7883 pada 7,14 ms, dibanding ResNet50 CBAM 84,46% / 0,8204 pada 13,45 ms. Parameter PneuNet sekitar 153 kali lebih sedikit.

VGG16 memiliki latency sedikit lebih rendah, tetapi recall NORMAL 0% dan macro-F1 0,3846. Karena itu klaim tercepat yang masih mengenali kedua kelas lebih tepat daripada klaim tercepat tanpa kualifikasi. ResNet50 pretrained CLAHE+aug tetap memiliki accuracy/F1 lebih tinggi daripada PneuNet pada kondisi tersebut.

Hasil satu seed/split belum menunjukkan keunggulan yang stabil atau validasi klinis. Detail ambiguitas arsitektur dan preprocessing terdapat pada [README](../README.md) dan [MODELING](../MODELING.md).
