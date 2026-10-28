Referensi terpilih untuk augmentasi dan resize chest X-ray

PDF sumber diunduh terpisah melalui DOI/tautan artikel; PDF tidak disertakan di Git. Semua nomor halaman di bawah merujuk pada urutan halaman PDF, termasuk halaman sampul jika ada. Referensi dipilih karena menguji augmentasi, menggunakan motion blur pada CXR, mengevaluasi kombinasi, atau menjelaskan padding untuk input CNN. Tidak ada paper dalam pilihan ini yang membuktikan satu distribusi motion blur : rotasi : kombinasi sebagai optimum untuk PneuNet.

1. Rainio & Klen (2024), Comparison of simple augmentation transformations for a convolutional neural network classifying medical images.

   [Artikel jurnal](https://doi.org/10.1007/s11760-024-02998-5)

   Bagian 2.5, PDF halaman 4: tiap eksperimen memakai satu jenis augmentasi; setiap gambar training dibuatkan satu versi augmentasi dan gambar asli tetap dipakai. Komposisi akhir menjadi 1:1 asli dan augmentasi. Mereka menguji rotasi acak -15 sampai +15 derajat dan Gaussian blur dengan sigma 0,5-1,5. Dataset mencakup Chest X-Ray Images (Pneumonia). Hasil dan diskusi pada halaman 5-7 mendukung Gaussian blur serta rotasi ringan pada konteks eksperimen tersebut. Gaussian blur berbeda dari motion blur; hasilnya tidak langsung menetapkan efektivitas motion blur di notebook.

2. Kuo et al. (2021), Recalibration of deep learning models for abnormality detection in smartphone-captured chest radiograph.

   [Artikel jurnal](https://doi.org/10.1038/s41746-021-00393-9)

   PDF halaman 8, Data augmentation dan Hyperparameter optimization: memasukkan motion blur dalam simulasi foto radiograf; probabilitas motion blur ditelusuri pada rentang 0,2-0,5, dengan motion intensity 5. Rentang ini adalah ruang pencarian parameter, bukan satu probabilitas optimum universal. Konteksnya adalah foto CXR melalui smartphone, sehingga penerapannya ke file radiograf digital pediatrik perlu validasi tersendiri.

3. Van der Sluijs et al. (2023), Exploring Image Augmentations for Siamese Representation Learning with Chest X-Rays.

   [Naskah penulis](https://arxiv.org/abs/2301.12636)

   Figur 2 dan PDF halaman 5-6 membandingkan operasi tunggal serta pasangan augmentasi pada CXR. Random resized cropping dengan perubahan brightness/contrast memberikan pasangan terbaik dalam setup mereka. Appendix C, PDF halaman 15, melaporkan bahwa penambahan operasi lain, termasuk motion blur, tidak mengungguli pasangan tersebut. Eksperimennya menggunakan self-supervised SimSiam dan evaluasi linear probing; kesimpulan ini tidak langsung menentukan kebijakan terbaik untuk supervised PneuNet.

4. Vieira et al. (2021), Detecting pulmonary diseases using deep features in X-ray images.

   [Artikel jurnal](https://doi.org/10.1016/j.patcog.2021.108081) | [Teks lengkap PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8193974/)

   Bagian 3.2 dan Figur 2, PDF halaman 5 (halaman artikel 4), memakai zero-padding agar gambar menjadi persegi tanpa mengubah proporsi, dengan input CNN termasuk 224x224. Mereka juga mengusulkan downsampling berbasis maximum window. Tabel 7, PDF halaman 9, dan diskusi halaman 11 membandingkan metode resize tersebut dengan bilinear interpolation. Eksperimen ini bukan ablation padding versus stretch dengan interpolasi yang identik; peningkatannya tidak boleh dianggap sebagai bukti bahwa letterbox saja selalu lebih akurat.

Keputusan implementasi pada notebook

- Resize diubah menjadi letterbox 224x224, dengan skala yang sama untuk kedua sumbu dan padding hitam terpusat. Dasarnya adalah pemeliharaan proporsi pada Vieira et al. (2021), bagian 3.2. Pilihan ini belum merupakan bukti akurasi lebih tinggi daripada direct resize karena paper tersebut juga mengubah metode downsampling.

- Training memakai kuota per kelas per epoch: 50% asli, 25% rotasi -15 sampai +15 derajat, dan 25% Gaussian blur dengan sigma 0,5-1,5. Pembulatan menghasilkan 2.323 asli, 1.162 rotasi, dan 1.161 Gaussian blur dari 4.646 gambar. Pada generator online, kategori diacak ulang setiap epoch; array ablation menyimpan satu realisasi tetap dengan seed 42. Setiap gambar dipakai satu kali per epoch. Grafik total dan per kelas memakai fungsi perencanaan yang sama dengan generator training.

- Rainio & Klen (2024), bagian 2.5, mendasari campuran asli-augmentasi 1:1 dan pilihan rotasi serta Gaussian blur. Pembagian bagian augmentasi menjadi dua porsi 25% adalah rancangan eksperimen notebook, bukan rasio optimum yang mereka buktikan. Skema offline 1 asli + 1 augmentasi menggandakan jumlah contoh; skema online notebook mempertahankan jumlah penyajian per epoch.

- [Athalye & Arnaout (2023), Domain-guided data augmentation for deep learning on medical imaging](https://doi.org/10.1371/journal.pone.0282532), PDF halaman 8, memberikan contoh campuran online sekitar 50:50 per kelas pada ultrasound. Perbedaan modalitas tetap membatasi penerapan langsung hasilnya ke pneumonia.

- Setiap gambar memperoleh satu kategori augmentasi. CLAHE adalah preprocessing tetap khusus jalur CLAHE, sebelum letterbox; baseline hanya memakai resize dan normalisasi. Val/test baseline memakai resize-normalisasi; X_val_clahe dan X_test_clahe memakai CLAHE-resize-normalisasi; parameter dipilih melalui validation set.

Alasan tidak menggunakan motion blur

1. **Bukti yang lebih langsung untuk tugas ini.** [Rainio & Klen (2024)](https://doi.org/10.1007/s11760-024-02998-5) menguji Gaussian blur dan rotasi pada data termasuk Chest X-Ray Images (Pneumonia). Ini mendasari prioritas kedua operasi untuk notebook ini. Gaussian blur tidak dianggap sebagai simulasi identik gerakan pasien.

2. **Konteks motion blur pada referensi berbeda.** [Kuo et al. (2021)](https://doi.org/10.1038/s41746-021-00393-9), PDF halaman 8, memakainya dalam simulasi foto CXR melalui smartphone. Notebook menggunakan file radiograf digital, sehingga hasil tersebut belum menentukan manfaat motion blur pada dataset ini.

3. **Menambah operasi belum tentu meningkatkan hasil.** [Van der Sluijs et al. (2023)](https://arxiv.org/abs/2301.12636), Appendix C, PDF halaman 15, melaporkan tambahan motion blur tidak mengungguli kombinasi terbaik mereka pada evaluasi self-supervised. Temuan ini mendukung pengujian operasi tambahan secara terpisah, bukan klaim bahwa motion blur selalu merugikan.

Pemilihan Gaussian blur dan penghapusan motion blur adalah keputusan eksperimen yang sesuai dengan bukti yang tersedia. Kekuatan blur, peluang augmentasi, dan perbandingan letterbox versus direct resize tetap perlu dievaluasi melalui validation set; test digunakan untuk evaluasi final.

PDF sumber telah diperiksa melalui signature, kecocokan judul, pembacaan halaman, dan render bukti utama. Notebook diperbarui untuk menjalankan keputusan di atas; file gambar dan pembagian folder dataset tidak diubah.

CLAHE sebagai preprocessing tetap dan persiapan ablation

- [Qiu et al. (2023), IRCM-Caps: An X-ray image detection method for COVID-19](https://doi.org/10.1111/crj.13599), bagian 3.2 dan Tabel 2 (PDF halaman 5), melaporkan akurasi 92,15% menjadi 93,04% dan AUC 0,9117 menjadi 0,9217 setelah CLAHE pada klasifikasi tiga kelas CXR. Parameter `clipLimit=2.5` dan `tileGridSize=(8,8)` menjadi titik awal tetap di notebook. .

- [Buriboev et al. (2025), Binary Classification of Pneumonia in Chest X-Ray Images Using Modified Contrast-Limited Adaptive Histogram Equalization Algorithm](https://doi.org/10.3390/s25133976), Tabel 5 (PDF halaman 10), melaporkan akurasi 86,4% pada data original dan 93,5% dengan CLAHE standar. Modified CLAHE pada paper merupakan metode lain dan tidak diimplementasikan di notebook. .

- Urutan pemrosesan tetap adalah grayscale -> CLAHE -> letterbox 224x224. CLAHE dilakukan sebelum padding agar area padding tambahan tidak memengaruhi histogram lokal. Tahap ini digunakan pada jalur CLAHE; baseline train/val/test hanya memakai letterbox dan normalisasi. Manfaatnya pada PneuNet tetap perlu diuji; hasil paper tidak menjamin peningkatan yang sama.

- `X_train` berisi baseline letterbox + normalisasi tanpa CLAHE. `X_train_clahe_aug` berisi CLAHE + letterbox + augmentasi 50:25:25 + normalisasi, dengan seed 42. Keduanya mempertahankan urutan sumber, memiliki bentuk `(4646,224,224)` dan dtype float32 dalam rentang 0-1. Label `y_train` dan `y_train_clahe_aug` identik. Manifest mencatat mode tiap gambar.

- `ablation_batches` mengacak kedua kondisi dengan seed yang sama dan menambahkan kanal; array float32 yang sudah 0-1 tidak dinormalisasi dua kali. `train_batches` tersedia terpisah untuk eksperimen augmentasi online. Grafik memakai mode yang benar-benar membentuk array augmentasi. Istilah asli pada grafik berarti tanpa augmentasi acak setelah preprocessing tetap.

Pemisahan baseline dan jalur CLAHE

- Nama array kanonis: `X_train` dan `X_train_clahe_aug`. Nama lama tidak digunakan dalam kode atau ekspor baru.
- `X_train` hanya memakai letterbox 224x224 dan normalisasi 0-1. CLAHE hanya menjadi bagian jalur kedua.
- Pada `X_train_clahe_aug`, seluruh gambar mendapat CLAHE; kuota 50:25:25 berarti CLAHE saja, CLAHE + rotasi, dan CLAHE + Gaussian blur.
- Gunakan `X_val`/`X_test` untuk baseline dan `X_val_clahe`/`X_test_clahe` untuk jalur CLAHE, dengan label evaluasi yang sama dan tanpa augmentasi acak.
- Perbandingan dua kondisi mengukur perubahan gabungan CLAHE dan augmentasi, bukan efek salah satu komponen secara terpisah.
- Array ternormalisasi disimpan sebagai memory-map NumPy di direktori cache yang dicetak notebook agar pemrosesan tidak membutuhkan seluruh array sekaligus di RAM.
