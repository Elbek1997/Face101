from library.face_detection import *
from library.face_identification import *

from os import listdir

import time
# Face detector
face_detector = Face_Detector()

# Face identification
face_identification = Face_Identification()

names = ["dinesh", "erlic", "gilfoyle", "richard", "jared"]

database = [ face_identification.generate_embeddings(cv2.imread("database/%s.png" % name))  for name in names]

test_image = cv2.imread("test.jpg")

faces = face_detector.detect_image(test_image)

for face in faces:

    embedding = face_identification.generate_embeddings(face)
    index = np.argmin([pairwise_distance(embedding, emb) for emb in database])
    
    print("name: %s:, distance: %f" %
          (names[index], 100*pairwise_distance(embedding, database[index])))





    







