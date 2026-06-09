# Face Attendance Project

Project ini memakai `face_recognition` dan `OpenCV` untuk mendeteksi wajah dari webcam dan mencatat kehadiran ke `Attendance.csv`.

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

- Tekan `q` untuk menutup webcam
- Wajah yang cocok dengan data training akan diberi nama
- Wajah yang tidak cocok akan dilabeli `Unknown`

## Folder dan File Penting

- `ImagesAttendance/` untuk gambar wajah referensi attendance
- `ImageBasic/` untuk gambar latihan pada `basic.py`
- `Attendance.csv` untuk hasil absensi

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
