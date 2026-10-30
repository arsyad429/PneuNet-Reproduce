# Modeling dan reproduksi

Panduan ini berlaku untuk tiga notebook pada [README](README.md). Semua kode model dan runner berada di notebook; file dataset, bobot, cache, dan output tidak dilacak Git.

## Varian arsitektur

| Varian | SE / attention | Learnable Pooling | Parameter |
|---|---|---|---:|
| PneuNet v1 | SE(128), setelah concat empat cabang | Conv 128→1, BN satu kanal, sigmoid, weighted sum | 142.412 |
| PneuNet v2 | SE(32) pada tiga cabang depthwise-separable; tanpa SE global | Conv 128→128 + ReLU, Conv 128→1 + sigmoid, weighted sum; tanpa BN | 157.224 |
| ResNet50 + SE | SE pada feature map akhir | Global average pooling | 24.036.545 |
| ResNet50 + CBAM | Channel dan spatial attention pada feature map akhir | Global average pooling | 24.034.467 |
| ResNet50 pretrained | Tanpa attention tambahan | Global average pooling | 23.510.081 |

PneuNet menggabungkan satu cabang pointwise dengan tiga cabang depthwise-pointwise. Setelah downsampling, ASPP memakai 1 x 1 dan atrous rates 1, 3, 6. Keluaran model adalah logit; sigmoid dipakai untuk probabilitas pelaporan.

BN setelah stem mengikuti uraian paper mengenai penggunaan BN pada convolution. Urutan Conv-BN-ReLU, epsilon Adam/BN, stride/padding, dan detail backbone Torchvision merupakan pilihan implementasi ketika paper tidak lengkap. Total parameter aktual dicatat; tidak diasumsikan sama dengan 1,84 juta pada paper.

## Protokol training

| Pengaturan | Nilai full |
|---|---|
| Seed | 42 |
| Input | Grayscale 224 x 224; diulang menjadi 3 kanal di model |
| Optimizer | Adam, LR tetap 0,001; beta 0,9/0,999; epsilon 1e-7 |
| Loss | BCEWithLogitsLoss + explicit L2 pada Conv2d/Linear |
| L2 | 0,001 x sum(weight squared); Adam weight_decay=0 |
| Effective batch / micro-batch | 32 / 4 |
| Epoch / early stopping | Maksimal 50; patience 7 berdasarkan validation loss |
| Dropout | 0,3 pada lokasi implementasi |
| Precision | FP32, tanpa AMP/TF32 |
| Threshold klasifikasi | 0,5 |
| Bobot awal | None, kecuali notebook ResNet50 pretrained ImageNet-1K V2 |

Gradient accumulation menghasilkan effective batch 32, tetapi statistik BatchNorm memakai micro-batch 4. Ini berbeda dari physical batch 32. Pilih micro-batch 32 hanya jika kapasitas GPU mencukupi.

Normalisasi ImageNet dilakukan sekali di dalam adapter ResNet50 pretrained. Model dari awal memakai input [0,1]. Tidak ada pembagian 255 kedua atau augmentasi acak tambahan pada DataLoader.

## Tiga alur eksekusi

| Alur | RUN_TRAINING | REBUILD_PREPROCESSING | MODELING_RUN_MODE |
|---|---|---|---|
| Laporan dari artefak lokal | False | False | Tidak memulai training |
| Preprocessing pertama + smoke | True | True | smoke |
| Full setelah cache tersedia | True | False | full |

Konfigurasi saat ini adalah `RUN_TRAINING=True`, `REBUILD_PREPROCESSING=True`, dan mode `full`. Periksa pengaturan sebelum Run All. Untuk laporan saja gunakan kedua flag `False`; untuk smoke gunakan mode `smoke`. Cell laporan full dilewati saat smoke atau saat belum ada run full lengkap.

Smoke menggunakan dua citra train dan dua validation, satu epoch dan satu update optimizer. Nilai smoke hanya memeriksa alur kode, bukan performa model. Full menyimpan checkpoint terbaik lalu mengevaluasi train, validation, dan test. Runner menempatkan run baru di `output_main/modeling/`, `output_resnet_cbam/modeling/`, atau `output_resnet50/modeling/` sesuai notebook, dengan subfolder run bertimestamp.

Untuk full training semua pembanding, pertahankan `SELECTED_MODELS = list(MODEL_NAMES)` dan kedua kondisi. Pembaca laporan mengharapkan semua model/kondisi notebook tersedia; jika menjalankan subset, selaraskan konfigurasi laporan dengan artefak yang benar-benar ada.

## Membaca laporan tanpa training

`SAVED_RUN_DIR = MODELING_RESULTS_ROOT` memilih run full lengkap terbaru secara otomatis. Variabel ini juga menerima path satu folder run full untuk pemilihan manual. Pembaca memeriksa status COMPLETE, identitas model/kondisi, versi arsitektur bila tersedia, history, dan kecocokan metrik terhadap prediksi.

Bagian **Confusion matrix dan classification report seluruh eksperimen** memiliki tiga cell mandiri. Pada kernel baru, fallback memilih run full lengkap terbaru dari folder modeling notebook terkait. Pembaca laporan hanya membaca CSV dan `result.json`; pemilihan run memeriksa keberadaan history, prediksi semua split, dan `best.pt`. Jika belum ada run lengkap, laporan dilewati dengan pesan informasi. Pada run full yang baru selesai di kernel yang sama, laporan memakai `modeling_output_dir` dari runner.

Urutan kelas: NORMAL=0, PNEUMONIA=1. Matrix berbentuk `[[TN, FP], [FN, TP]]`. Prediksi tersimpan dipakai untuk menghitung precision, recall, F1, accuracy, dan support; nilainya dicocokkan dengan `result.json`. Warna matrix dinormalisasi per kelas aktual, sedangkan anotasi memuat hitungan dan persentase.

`CLASSIFICATION_REPORT_SPLITS = ("train", "val", "test")` menampilkan seluruh split. Test menjadi acuan performa generalisasi; train dan validation bukan out-of-fold. Dengan `CLASSIFICATION_REPORT_EXPORT=True`, tabel agregat dan figure disimpan di `<OUTPUT_ROOT>/notebook_reports/classification_reports/`.

## Grad-CAM dan latency

Grad-CAM memerlukan checkpoint `best.pt` yang cocok, input preprocessing yang sama, serta PyTorch CUDA. Metode ini memakai gradien terhadap feature map; tidak memanggil optimizer. Target NORMAL memakai negatif logit PNEUMONIA. Heatmap nol tetap ditandai dan kontribusi spasial bukan segmentasi lesi.

Tabel benchmark pada notebook membaca CSV yang telah tersedia. Persiapan repository tidak menjalankan ulang benchmark. Protokol dan batas latency historis dijelaskan pada [hasil eksperimen](docs/results.md).

## Status publikasi

Output notebook dikosongkan agar tidak mengunggah citra. Markdown abstrak/kesimpulan dan `docs/experiment_results.csv` merangkum run full terdahulu. Salinan publikasi hanya mengubah portabilitas path/interpreter, membersihkan output/metadata lokal, dan membatasi laporan full saat smoke; definisi arsitektur serta perhitungan training dipertahankan. Tidak ada full training baru saat persiapan repository.
