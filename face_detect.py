import cv2
import mediapipe as mp

def detect_faces_in_image(image_path):
    # Initialize Mediapipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    # Read the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    # Convert the image from BGR to RGB (Mediapipe uses RGB format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Create a face detection object
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    # Process the image
    results = face_detection.process(image_rgb)
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Display the output image
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = 'cropped320.jpg'
    detect_faces_in_image(image_path)
