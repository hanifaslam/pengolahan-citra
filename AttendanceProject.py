from pathlib import Path
from datetime import datetime

import cv2
import face_recognition
import numpy as np

# from PIL import ImageGrab

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "ImagesAttendance"
ATTENDANCE_FILE = BASE_DIR / "Attendance.csv"

images = []
classNames = []

myList = [file.name for file in IMAGE_DIR.iterdir() if file.is_file()]
print(myList)

for cl in myList:
    curImg = cv2.imread(str(IMAGE_DIR / cl))
    if curImg is None:
        continue
    images.append(curImg)
    classNames.append(Path(cl).stem)

print(classNames)


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
    ATTENDANCE_FILE.touch(exist_ok=True)

    with open(ATTENDANCE_FILE, "r+", encoding="utf-8") as f:
        myDataList = f.readlines()
        nameList = []

        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{dtString}")


#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown, classNames = findEncodings(images, classNames)
print("Encoding Complete")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        continue

    # img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        if not encodeListKnown:
            continue

        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(
                img,
                name,
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            markAttendance(name)

    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
