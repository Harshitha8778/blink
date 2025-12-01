import cv2
cap = cv2.VideoCapture(0)  # try 0, 1, or 2
if cap.isOpened():
    print("Camera works!")
else:
    print("Cannot open camera")
cap.release()
