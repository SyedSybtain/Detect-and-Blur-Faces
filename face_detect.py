import cv2
import mediapipe as mp
import numpy as np

img = cv2.imread("faces10.jpg")
faces = 0
imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
print(imgRGB.shape)
mpFACE = mp.solutions.face_detection
face_detection = mpFACE.FaceDetection()
results = face_detection.process(imgRGB)
if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle( img, (x, y), (x+w, y+h), (0,255,0), 4)
            faces = faces + 1


string = str(faces)+'faces in image'
cv2.putText(img, string ,(50,50),1,3,(0,0,255),4)
# Display the output image
print(faces)
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
