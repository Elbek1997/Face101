import numpy as np
import cv2


def pre_process(img):

    # Width and height
    w = 60
    h = 60

    # Convert BGR to Gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize image
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

    # Convert to Blob
    img = cv2.dnn.blobFromImage(img)

    # Subtract mean and divide by std
    img = (img - np.mean(img)) / (0.000001 + np.std(img))

    print(img.shape)

    return img


def draw_points(points, img):

    # Get width and height
    height, width, _ = img.shape

    # Draw red dot on every point
    for i in range(len(points)//2):
        x = points[2*i] * width
        y = points[2*i+1] * height


        if i>=36 and i<=41:
            cv2.circle(img, (int(x), int(y)), 1, (255, 0, 0), 2)
        elif i>=42 and i<=47:
            cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), 2)
        else:
            cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 2)

    return img



# Load model
model = cv2.dnn.readNetFromCaffe("landmark_deploy.prototxt", "VanFace.caffemodel")

# Load image
img = cv2.imread("sample.png")

model.setInput(pre_process(img))

points = model.forward()[0]

img = draw_points(points, img)

cv2.imwrite("result.png", img)




