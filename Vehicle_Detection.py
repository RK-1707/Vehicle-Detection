import cv2
import numpy as np
import imutils
from datetime import datetime, timedelta

vs = cv2.VideoCapture("../assets/video/pivideo.mp4")
bgs = cv2.createBackgroundSubtractorKNN(history=30)

totalFrames = 0
start = datetime.now()

while True:
    ret, frame = vs.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask = bgs.apply(gray_frame)

    _, thresh = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
    threshDil = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(threshDil.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < 400:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "{:.2f}".format(cv2.contourArea(c)), (x - 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 0, 255), 1)

    totalFrames += 1

    frame = imutils.resize(frame, width=800)
    cv2.imshow("mask", mask)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

end = datetime.now()
diff = end - start
print("FPS: {:.2f}".format(totalFrames / (end - start).total_seconds()))
vs.release()
cv2.destroyAllWindows()
