import cv2

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(0)  # Try default camera (index 0)

if not cap.isOpened():
    print("Error: Camera is not accessible.")
    exit()
else:
    print("Camera is successfully opened.")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Blur each detected face
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
        frame[y:y+h, x:x+w] = blurred_face

    # Display the result
    cv2.imshow('Face Blur Tool', frame)

    # Exit the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
