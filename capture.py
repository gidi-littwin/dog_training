import cv2

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
cap.set(cv2.CAP_PROP_MODE, 150)

while cap.isOpened():
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if success:
        cv2.rectangle(gray, (0, 0), (510, 128), (0, 255, 0), 3)
        cv2.imshow("Result", gray)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
