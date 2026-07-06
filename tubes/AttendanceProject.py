from pathlib import Path
from datetime import datetime
import cv2
import face_recognition
import numpy as np
import tkinter as tk
from tkinter import simpledialog


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "ImagesAttendance"
MATCH_THRESHOLD = 0.50

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

images = []
classNames = []
encodeListKnown = []

def load_images_and_encodings():
    """Load semua gambar di folder dan buat encodings."""
    global images, classNames, encodeListKnown
    images.clear()
    classNames.clear()
    encodeListKnown.clear()
    
    myList = [file.name for file in IMAGE_DIR.iterdir() if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    print(f"Loading images: {myList}")
    
    for cl in myList:
        curImg = cv2.imread(str(IMAGE_DIR / cl))
        if curImg is None:
            continue
        images.append(curImg)
        classNames.append(Path(cl).stem)
        
    encodeListKnown, classNames = findEncodings(images, classNames)
    print("Encoding Selesai! Siap mendeteksi wajah.")

def findEncodings(images, classNames):
    encodeList = []
    validClassNames = []
    
    for img, className in zip(images, classNames):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            encodeList.append(encodings[0])
            validClassNames.append(className)
            
    return encodeList, validClassNames

def markAttendance(name):
    """Mencatat kehadiran di file CSV harian."""
    now = datetime.now()
    dateString = now.strftime("%Y-%m-%d")
    fileName = BASE_DIR / f"Attendance_{dateString}.csv"
    
    if not fileName.exists():
        with open(fileName, "w", encoding="utf-8") as f:
            f.write("Nama,Waktu\n")
            
    with open(fileName, "r+", encoding="utf-8") as f:
        myDataList = f.readlines()
        nameList = [line.split(",")[0].strip() for line in myDataList]
        
        if name not in nameList:
            timeString = now.strftime("%H:%M:%S")
            if myDataList and not myDataList[-1].endswith("\n"):
                f.write("\n")
            f.write(f"{name},{timeString}\n")
            print(f"[{timeString}] Kehadiran dicatat untuk {name}")

def register_new_face(img):
    """Membuka dialog popup untuk mendaftarkan wajah baru."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    name = simpledialog.askstring("Registrasi Wajah Baru", "Masukkan nama untuk wajah ini:")
    root.destroy()
    
    if name:
        name = name.strip()
        if name:
            save_path = IMAGE_DIR / f"{name}.jpg"
            cv2.imwrite(str(save_path), img)
            print(f"Berhasil menyimpan wajah baru: {name}")
            
            print("Memuat ulang database wajah...")
            load_images_and_encodings()
    else:
        print("Registrasi dibatalkan.")

load_images_and_encodings()

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        continue
        
    img = cv2.flip(img, 1)

    clean_img = img.copy()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        name = "Unknown"
        if encodeListKnown:
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            if len(faceDis) > 0:
                matchIndex = np.argmin(faceDis)
                if faceDis[matchIndex] < MATCH_THRESHOLD:
                    name = classNames[matchIndex].upper()
                    markAttendance(name)
        
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        boxColor = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), boxColor, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), boxColor, cv2.FILLED)
        cv2.putText(
            img,
            name,
            (x1 + 6, y2 - 6),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        
    cv2.putText(img, "Tekan 'r' : Registrasi Wajah", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "Tekan 'q' : Keluar", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("Webcam", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        register_new_face(clean_img)

cap.release()
cv2.destroyAllWindows()
