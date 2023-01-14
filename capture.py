import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision.models as models
from torchvision import transforms
from imagenet_classes import CLASSES


def draw_indication(img):
    cv2.circle(img, (8, 8), 5, (0, 0, 255), -1)


frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
cap.set(cv2.CAP_PROP_MODE, 150)

# Set frame rate
# cap.set(cv2.CAP_PROP_FPS, 10)
# fps = int(cap.get(5))
# print("fps:", fps)


def transform_image(img_cv):
    # Normalization transformation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    img_cv = np.transpose(img_cv, (2, 0, 1))
    img_pil = transform(img_cv)
    img_pil = torch.unsqueeze(img_pil, 0)
    return img_pil


# Load model
model = models.resnet18(pretrained=True)
_ = model.eval()  # Put model in eval mode

prev_gray = None
diff_array = list(np.zeros(30))
while cap.isOpened():
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        diff = np.sum((gray - prev_gray)**2)/frameWidth/frameHeight
        diff_array.append(diff)
        diff_array.pop(0)
        diff_move_avg = np.mean(diff_array)
        print(f'Image diff is: {diff_move_avg}')
        movement_detected = False
        if diff_move_avg > 3.:
            movement_detected = True

        if success:
            if movement_detected:
                draw_indication(img)
                img_pil = transform_image(img)
                out = model(img_pil).detach().numpy()
                max_idx = np.argmax(out)

                print(CLASSES[max_idx])

            cv2.imshow("Result", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    prev_gray = gray


cap.release()
cv2.destroyAllWindows()

# plt.figure(1)
# plt.imshow(img)
# plt.show()
