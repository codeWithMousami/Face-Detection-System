#haarcascade_frontalface_alt.xml

import cv2
cascade =cv2.CascadeClassifier(cv2.haarcascades+'haarcascade_frontalface_alt.xml')
cap =cv2.VideoCapture('Video.mp4')

while True:
    ret,frame =cap.read()
    gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('Face Detection ',frame)
    k = cv2.waitKey(30)
    if k == 27:
        break
cap.release() #to stop the camera
cv2.destroyWindow()