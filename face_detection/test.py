import time

import cv2
import numpy as np


# Set opencl
cv2.ocl.setUseOpenCL(True)

#region Detector
protoPath = "deploy.prototxt"
modelPath = "res10_300x300_ssd_iter_140000.caffemodel"

detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
#endregion

# Load test.jpg
image = cv2.imread("test.jpg")


imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300), 1.0 ))

start_time = time.time()


detector.setInput(imageBlob)

detections = detector.forward()

measure = time.time() - start_time

print("Measured time:", measure)
