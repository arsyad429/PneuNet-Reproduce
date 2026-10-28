# PneuNet Reproduce

Reproduksi dan eksperimen ablation **PneuNet** untuk klasifikasi **NORMAL vs PNEUMONIA** pada chest X-ray menggunakan **PyTorch dan GPU NVIDIA**. Fokus proyek adalah menilai keseimbangan performa klasifikasi, jumlah parameter, dan latency inferensi sebuah CNN ringan.

Rujukan utama: [Saranyaraj et al. (2025), *PneuNet: a lightweight convolutional neural network with multiscale feature fusion for automated pneumonia detection from chest X-rays*](https://doi.org/10.3389/fmed.2025.1713587).

Repository berisi **kode notebook, dokumentasi, dependensi, dan ringkasan numerik hasil**. Dataset, citra, checkpoint/bobot model, array preprocessing, PDF paper, serta output notebook yang dapat memuat X-ray **tidak disertakan**. Notebook publik telah dibersihkan dari output dan attachment; notebook kerja lokal tidak diubah.

## Isi repository

| Notebook | Model | Inisialisasi | Eksperimen |
|---|---|---|---:|
| [Ablation utama](reproduce-kc_main_ablation.ipynb) | PneuNet v1, ResNet50 + SE, DenseNet121, EfficientNetB4, VGG16 | Dari awal | 5 model x 2 kondisi = 10 |
| [PneuNet revisi dan ResNet50 CBAM](reproduce_pneunet_resnet50_cbam.ipynb) | PneuNet v2, ResNet50 + CBAM | Dari awal | 2 model x 2 kondisi = 4 |
| [ResNet50 pretrained](reproduce_resnet50_pretrained.ipynb) | ResNet50 tanpa attention tambahan | ImageNet-1K V2; fine-tuning seluruh layer | 1 model x 2 kondisi = 2 |

Seluruh definisi arsitektur, Dataset/DataLoader, training loop, evaluasi, dan Grad-CAM berada **di dalam notebook**. Tidak ada modul Python model terpisah. Versi Kaggle lama dan notebook contoh dari proyek lain tidak dimasukkan ke publikasi ini.

```text
PneuNet-Reproduce/
├── README.md
├── MODELING.md
├── reproduce-kc_main_ablation.ipynb
├── reproduce_pneunet_resnet50_cbam.ipynb
├── reproduce_resnet50_pretrained.ipynb
├── requirements-pneunet.txt
├── requirements-pneunet-lock.txt
├── docs/
│   ├── results.md
│   └── experiment_results.csv
└── Paper/
    └── Catatan referensi augmentasi dan resize.md
```

## Lingkungan

Eksperimen historis memakai Python **3.10.20**, PyTorch **2.11.0+cu128**, Torchvision **0.26.0+cu128**, dan **NVIDIA GeForce RTX 3060 Laptop GPU, VRAM 6 GB**, dalam FP32. Versi paket dicatat pada [requirements](requirements-pneunet.txt).

Contoh menyiapkan environment baru:

```bash
git clone https://github.com/arsyad429/PneuNet-Reproduce.git
cd PneuNet-Reproduce
conda create -n pneunet python=3.10.20
conda activate pneunet
python -m pip install -r requirements-pneunet.txt
python -m ipykernel install --user --name pneunet --display-name "PneuNet (PyTorch CUDA)"
python -m jupyterlab
```

Jika environment CUDA sudah tersedia, cukup gunakan environment tersebut dan daftarkan kernelnya. Pilih kernel **PneuNet (PyTorch CUDA)** pada notebook. Driver NVIDIA harus mendukung build CUDA PyTorch yang dipasang. Notebook memeriksa CUDA sebelum modeling; laporan dari CSV dapat dijalankan terpisah di CPU.

Mulai Jupyter dari root repository. Pada salinan publikasi, `WORKSPACE = Path.cwd().resolve()`; path tidak dikunci ke drive Windows atau interpreter komputer penulis.

## Dataset dan pembagian data

Siapkan chest X-ray secara terpisah, misalnya dari [dataset Kaggle proyek](https://www.kaggle.com/datasets/xii2muhammadarsyad/pneumonia-datasets), kemudian tempatkan sesuai struktur berikut. Dataset tidak diunduh oleh kode repository ini.

```text
Datasets/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Train | 1.198 | 3.448 | 4.646 |
| Validation | 151 | 435 | 586 |
| Test | 234 | 390 | 624 |
| **Total** | **1.583** | **4.273** | **5.856** |

Notebook mengikuti **folder fisik yang telah disiapkan**, tanpa membagi ulang test set secara otomatis. Pembagian validation bawaan suatu unduhan belum tentu sama dengan proyek ini. Untuk mereplikasi angka historis dibutuhkan pembagian citra yang sama, bukan sekadar jumlah citra yang sama. Dataset dan manifest per citra tidak diterbitkan di repository ini.

## Dua kondisi ablation

| Kondisi | Train | Validation dan test |
|---|---|---|
| **Baseline** | Grayscale, letterbox 224 x 224, normalisasi [0,1] | Sama, tanpa augmentasi |
| **CLAHE + augmentasi** | CLAHE, letterbox, satu kategori augmentasi tetap per citra, normalisasi [0,1] | CLAHE, letterbox, normalisasi; tanpa augmentasi acak |

Distribusi train pada cabang kedua: **50% CLAHE saja, 25% CLAHE + rotasi, 25% CLAHE + Gaussian blur**. CLAHE diterapkan pada seluruh citra cabang tersebut. Rotasi dibatasi ±15 derajat dan sigma blur 0,5–1,5; CLAHE memakai clip limit 2,5 dan grid 8 x 8. Kuota dijaga per kelas dan realisasi augmentasi array menggunakan seed 42.

Rasio 50:25:25 merupakan **rancangan eksperimen proyek**, bukan rasio optimum universal yang dibuktikan paper. Ablation ini mengubah CLAHE dan augmentasi sekaligus; kontribusi keduanya belum terpisah. [Catatan referensi](<Paper/Catatan referensi augmentasi dan resize.md>) menjelaskan Gaussian blur, alasan tidak memakai motion blur, CLAHE, dan letterbox.

## Cara menjalankan

Semua pengaturan berada pada **cell kode pertama**. Alur notebook: load data → EDA → preprocessing dan augmentasi → modeling dan evaluasi → error analysis dengan Grad-CAM.

Nilai awal `RUN_TRAINING=False` mencegah training otomatis. Dataset, cache, dan artefak full tidak ada pada fresh clone; **pilih salah satu alur di bawah sebelum menjalankan seluruh notebook**. Mode laporan tidak akan mengunduh atau membuat artefak yang belum tersedia.

### 1. Preprocessing dan smoke test

Setelah dataset tersedia, ubah konfigurasi:

```python
RUN_TRAINING = True
REBUILD_PREPROCESSING = True  # Hanya untuk pembuatan cache pertama.
MODELING_RUN_MODE = "smoke"
SELECTED_MODELS = list(MODEL_NAMES)
SELECTED_CONDITIONS = ["baseline", "clahe_aug"]
```

Jalankan cell berurutan. Smoke memakai dua citra train dan dua validation, satu epoch, dan satu langkah optimizer per eksperimen. **Smoke bukan pengukuran performa** dan tidak mengevaluasi test set. Cell laporan full dilewati pada mode smoke. Preprocessing tetap menyiapkan array split lengkap, sehingga memerlukan ruang disk di luar bobot model.

### 2. Full training

Setelah cache tersedia:

```python
RUN_TRAINING = True
REBUILD_PREPROCESSING = False
MODELING_RUN_MODE = "full"
```

Training berjalan maksimal 50 epoch, memakai Adam LR 0,001, effective batch 32 dengan micro-batch 4, dropout 0,3, explicit L2 0,001, dan early stopping patience 7 berdasarkan validation loss. Checkpoint terbaik dipilih sebelum evaluasi test. Detail ada pada [MODELING.md](MODELING.md).

Artefak baru disimpan ke `outputs/modeling/pytorch_full_.../` dan diabaikan Git. **Full training hanya berjalan jika diaktifkan sendiri oleh pengguna.** Pembuatan repository ini tidak menjalankan ulang training.

### 3. Laporan dari eksperimen yang sudah selesai

```python
RUN_TRAINING = False
REBUILD_PREPROCESSING = False
SAVED_RUN_DIR = WORKSPACE / "outputs" / "modeling" / "nama_run_full_anda"
```

Folder run harus memuat `{model}__{baseline|clahe_aug}/`, dengan struktur:

```text
nama_run_full_anda/
├── PneuNet__baseline/
│   ├── result.json
│   ├── full_history.csv
│   ├── train_predictions.csv
│   ├── val_predictions.csv
│   ├── test_predictions.csv
│   └── best.pt
└── PneuNet__clahe_aug/
    └── ...
```

Semua model yang tercantum pada `MODEL_NAMES` perlu tersedia dalam run notebook yang dipilih. `SAVED_RUN_DIR` awal memakai placeholder `outputs/modeling/saved_full_v1`, `saved_full_v2`, atau `saved_full_v3`; ganti dengan lokasi artefak Anda.

- **Confusion matrix dan classification report:** jalankan hanya tiga cell pada bagian tersebut. CSV prediksi dan `result.json` cukup; tidak memuat model, checkpoint, atau citra. `CLASSIFICATION_REPORT_SPLITS = ("train", "val", "test")` menampilkan seluruh split; gunakan `("test",)` bila hanya ingin test.
- **Kurva train/validation loss:** dibaca dari `full_history.csv`.
- **Grad-CAM:** memerlukan `best.pt`, dataset/cache yang sesuai, definisi model, dan GPU. Ini melakukan forward/backward untuk visualisasi, tanpa update optimizer.
- **Tabel latency:** membaca CSV benchmark yang telah tersedia; tidak menjalankan benchmark otomatis. Ringkasan historis yang diterbitkan tersedia pada `docs/experiment_results.csv`.

Output laporan tersimpan di `outputs/notebook_reports/`. Untuk keseluruhan 16 eksperimen pada tiga split, tersedia 48 confusion matrix dan 48 classification report setelah artefak lokal disediakan. Angka laporan diperiksa kembali terhadap `result.json`.

## Ringkasan hasil historis

Hasil di bawah berasal dari eksperimen full lokal terdahulu pada 624 citra test. Nilai tersebut tetap dicantumkan di markdown notebook meskipun output notebook telah dikosongkan. Seluruh 16 baris hasil tersedia pada [hasil lengkap](docs/results.md) dan [CSV agregat](docs/experiment_results.csv).

| Model | Parameter | Accuracy baseline | F1 macro baseline | Accuracy CLAHE+aug | F1 macro CLAHE+aug | GPU ms/citra CLAHE+aug |
|---|---:|---:|---:|---:|---:|---:|
| PneuNet v1 | 142.412 | 76,44% | 0,6930 | 82,37% | 0,7867 | 5,30 |
| ResNet50 + SE | 24.036.545 | 76,76% | 0,6996 | 78,21% | 0,7271 | 10,49 |
| DenseNet121 | 6.954.881 | 80,61% | 0,7594 | 85,42% | 0,8291 | 25,79 |
| EfficientNetB4 | 17.550.409 | 90,38% | 0,8928 | 84,62% | 0,8189 | 28,57 |
| VGG16 | 14.715.201 | 62,50% | 0,3846 | 62,50% | 0,3846 | 5,16 |
| PneuNet v2 | 157.224 | 82,37% | 0,7860 | 82,53% | 0,7883 | 7,14 |
| ResNet50 + CBAM | 24.034.467 | 77,56% | 0,7138 | 84,46% | 0,8204 | 13,45 |
| ResNet50 pretrained | 23.510.081 | 80,13% | 0,7555 | 83,65% | 0,8093 | 11,54 |

Pada CLAHE+augmentasi, PneuNet v1 memiliki latency sekitar **4,87 kali lebih cepat** dan sekitar **49 kali lebih sedikit parameter** daripada DenseNet121, dengan accuracy lebih rendah **3,04 poin persentase**. PneuNet v2 sekitar **1,88 kali lebih cepat** daripada ResNet50 CBAM pada run yang sama, dengan selisih accuracy **1,92 poin persentase**. Ini mendukung pembahasan trade-off efisiensi; tidak membuktikan keunggulan pada semua metrik.

**Konteks latency:** RTX 3060 Laptop GPU, FP32, input 224 x 224, batch 1, warm-up 20. GPU mean hanya mengukur forward; preprocessing dan transfer CPU ke GPU tidak termasuk. Paper mengukur Raspberry Pi 4 CPU dengan TensorFlow Lite, sehingga angka GPU proyek tidak dapat dibandingkan langsung dengan 20 ms pada paper. Pengukuran antar-run juga bersifat deskriptif.

## Batas reproduksi

- Teks dan tabel paper berbeda mengenai posisi SE serta detail learnable pooling. **v1** memakai SE setelah concatenation; **v2** memakai SE pada tiga cabang depthwise dan pooling dua proyeksi 128→128→1 tanpa BN. Proyeksi tambahan merupakan hipotesis rekonstruksi untuk mencocokkan 16.641 parameter pooling.
- Parameter PneuNet implementasi ini 0,142/0,157 juta, berbeda dari klaim paper sekitar 1,84 juta. Perbedaan arsitektur/framework dan BatchNorm micro-batch perlu diperhitungkan.
- Hanya ResNet50 pada notebook pretrained menggunakan bobot pretrained. Model v1 dan ResNet50 CBAM dilatih dari awal; **ViT tidak diuji**.
- VGG16 sedikit lebih cepat, tetapi recall NORMAL = 0% pada kedua kondisi. Kecepatan saja tidak cukup untuk menyimpulkan kualitas klasifikasi.
- Hasil berasal dari satu seed dan split fisik per gambar. EDA menemukan indikasi kelompok sumber beririsan pada train/validation dari nama file; identitas pasien belum terverifikasi. Tidak ada klaim generalisasi ke data eksternal atau validasi klinis.

## Berkas yang tidak diterbitkan

`.gitignore` mengecualikan dataset/citra, `.npy/.npz`, `.pt/.pth/.ckpt`, format model lain, folder `outputs*`, `tmp`, environment lokal, arsip, serta PDF/PPTX. Sebelum mengirim perubahan notebook, kosongkan **outputs dan attachments** agar citra atau data per sampel tidak ikut masuk ke Git.

## Referensi

- [Paper PneuNet](https://doi.org/10.3389/fmed.2025.1713587).
- [Referensi augmentasi, CLAHE, resize, dan alasan tidak menggunakan motion blur](<Paper/Catatan referensi augmentasi dan resize.md>).
- [Grad-CAM, Selvaraju et al.](https://arxiv.org/abs/1610.02391).
- [Torchvision ResNet50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).

PDF paper dan dataset diperoleh terpisah dari sumber masing-masing; bobot pretrained Torchvision dapat diunduh oleh Torchvision saat pengguna membangun model pretrained pertama kali.
