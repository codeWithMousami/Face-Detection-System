import cv2

# Load the Haar cascade classifier for frontal face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

# Load the video file
cap = cv2.VideoCapture('Video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the frame with face detection
    cv2.imshow('Face Detection', frame)

    # Check for the 'Esc' key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture object and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()
