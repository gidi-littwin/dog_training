import cv2
import numpy as np
from matplotlib import pyplot as plt

def draw_indication(image):
    cv2.circle(image, (8,8), 5, (0,0,255), -1)

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
cap.set(cv2.CAP_PROP_MODE, 150)

# Set frame rate
# cap.set(cv2.CAP_PROP_FPS, 10)
# fps = int(cap.get(5))
# print("fps:", fps)

prev_gray = None
diff_array = list(np.ones(30)*1e8)
while cap.isOpened():
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        diff = np.sum((gray - prev_gray)**2)/frameWidth/frameHeight
        diff_array.append(diff)
        diff_array.pop(0)

        diff_move_avg = np.mean(diff_array)

        print(f'Image diff is: {diff_move_avg}')

    prev_gray = gray

    if success:
        # cv2.circle(img, (8, 8), (510, 128), (0, 255, 0), -1)
        cv2.imshow("Result", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

plt.figure(1)
plt.imshow(img)
plt.show()
