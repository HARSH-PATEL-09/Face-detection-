import cv2

cap_video = cv2.VideoCapture(0)  # Use 0 for webcam or provide video path

if not cap_video.isOpened():
    print("Error: Could not open video source.")
    exit()

face_capture = cv2.CascadeClassifier("C:/Users/HARSH/AppData/Local/Programs/Python/Python313/Lib/site-packages/cv2/data/haarcascade_frontalface_default")

while True:
    ret, video_full = cap_video.read()

    if not ret or video_full is None:
        print("Error: Could not read frame from video.")
        break

    # Convert BGR to Grayscale (Correct Method)
    clr = cv2.cvtColor(video_full, cv2.COLOR_BGR2GRAY)

    face = face_capture.detectMultiScale(
        clr,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in face:
        cv2.rectangle(video_full, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", video_full)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_video.release()
cv2.destroyAllWindows()
