import time
import os
import cv2
import numpy as np
from argparse import ArgumentParser

from tqdm import tqdm

parser = ArgumentParser()

parser.add_argument("--folder", type=str)

args = parser.parse_args()


#region Detector
protoPath = "deploy.prototxt"
modelPath = "res10_300x300_ssd_iter_140000.caffemodel"

detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
#endregion

dirs = os.listdir(args.folder)

for inner in tqdm(dirs, desc="Dirs"):

    path = os.path.join(args.folder, inner)

    if not os.path.isdir(path):
        continue

    files = os.listdir(path)

    for f in tqdm(files, desc=path):

        file = os.path.join(path, f)

        image = cv2.imread(file)

        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300), 1.0 ))

        detector.setInput(imageBlob)

        detections = detector.forward()

        detection = detections[0][0][0]

        if detection[2]>=0.80:
            
            (h, w) = image.shape[:2]
            
            box = detection[3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # Minimum size
            if fH>=20 and fW>=20:
                cv2.imwrite(file, face)
            else:
                print("Face not found:", file)
                os.remove(file)

        else:

            print("Face not found:", file)
            os.remove(file)
