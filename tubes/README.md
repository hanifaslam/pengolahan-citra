# Face Attendance Project

Project ini memakai `face_recognition` dan `OpenCV` untuk mendeteksi wajah dari webcam dan mencatat kehadiran ke file CSV harian (contoh: `Attendance_YYYY-MM-DD.csv`).

## Menjalankan Project

Pastikan terminal ada di folder project ini:

```powershell
cd d:\Project-Coding\pengolahan-citra\tubes
```

Aktifkan virtual environment:

```powershell
.\.venv\Scripts\activate
```

Jalankan attendance webcam:

```powershell
python AttendanceProject.py
```

Jalankan file percobaan dasar:

```powershell
python basic.py
```

Kalau tidak ingin mengaktifkan `venv` secara manual, bisa langsung pakai:

```powershell
.\.venv\Scripts\python AttendanceProject.py
```

## Kontrol Webcam

- Tekan `r` untuk mendaftarkan wajah baru (akan muncul dialog untuk memasukkan nama, dan wajah otomatis tersimpan)
- Tekan `q` untuk menutup webcam
- Wajah yang cocok dengan data training akan diberi nama
- Wajah yang tidak cocok akan dilabeli `Unknown`

## Folder dan File Penting

- `ImagesAttendance/` untuk gambar wajah referensi attendance (gambar dari registrasi baru akan otomatis disimpan ke sini)
- `ImageBasic/` untuk gambar latihan pada `basic.py`
- `Attendance_YYYY-MM-DD.csv` untuk hasil absensi harian yang dicatat secara otomatis

## Install Ulang Dependency

Kalau virtual environment belum ada:

```powershell
python -m venv .venv
```

Lalu install dependency:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Status GPU

Saat ini project berjalan dengan `CPU`, bukan `GPU`.

Pengecekan terakhir:

```powershell
.\.venv\Scripts\python -c "import dlib; print(dlib.DLIB_USE_CUDA)"
```

Jika hasilnya `False`, berarti `dlib` masih belum memakai CUDA.
