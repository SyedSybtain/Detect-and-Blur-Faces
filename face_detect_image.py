import cv2
import mediapipe as mp
import numpy as np

img1 = cv2.imread("SYBTAIN.jpg")
img =cv2.resize(img1,(720,720))
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

            roi = img[y:y+h, x:x+w]
            # Apply Gaussian blur to the ROI
            blurred_section = cv2.GaussianBlur(roi, (25, 25), 100)
            # Replace the original rectangle with the blurred version
            img[y:y+h, x:x+w] = blurred_section
            

# Display the output image
print(faces)
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()