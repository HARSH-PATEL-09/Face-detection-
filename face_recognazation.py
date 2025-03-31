import cv2

face_capture = cv2.CascadeClassifier("C:/Users/HARSH/AppData/Local/Programs/Python/Python313/Lib/site-packages/cv2/data/haarcascade_frontalface_default")
cap_video = cv2.VideoCapture(0)

while True:
    ret,video_full = cap_video.read()
    clr = cv2.cvtColor(video_full,cv2.COLOR_BAYER_BG2GRAY)
    face = face_capture.detectMultiScale(
        clr,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flages=cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in face:
         cv2.rectangle(video_full,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("video_curr",video_full)
    if cv2.waitKey(10) == ord("m"):
        break
    cap_video.release()