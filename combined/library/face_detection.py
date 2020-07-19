import cv2
import numpy as np

# Face Detection 

class Face_Detector:

    def __init__(self, protoPath="models/face_detection/deploy.prototxt", 
        modelPath="models/face_detection/res10_300x300_ssd_iter_140000.caffemodel",
        threshold=0.8):

        self.threshold = threshold

        self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    

    def detect_image(self, image):
        """
        Detect image from given image

        returns: list of images
        """

        lst = []

        # Input image blob
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300), 1.0))

        self.detector.setInput(imageBlob)

        detections = self.detector.forward()

        # Image sizes
        (h, w) = image.shape[:2]

        # Loop through detections
        for detection in detections[0][0]:
            
            if detection[2]>=self.threshold:
                # Bounding box
                box = detection[3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # Minimum size
                if fH>=20 and fW>=20:
                    lst.append(face)

        return lst

    def detect_positions(self, image):
        """
        Detect image from given image

        returns: list of positions as (startX, startY, endX, endY)
        """

        lst = []

        # Input image blob
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300), 1.0))

        self.detector.setInput(imageBlob)

        detections = self.detector.forward()

        # Image sizes
        (h, w) = image.shape[:2]

        # Loop through detections
        for detection in detections[0][0]:

            if detection[2] >= self.threshold:
                # Bounding box
                box = detection[3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (fH, fW) = (endY - startY, endX - startX)

                # Minimum size
                if fH >= 20 and fW >= 20:
                    lst.append((startX, startY, endX, endY))

        return lst

