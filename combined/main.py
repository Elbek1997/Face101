from library.face_detection import *
from library.face_identification import *

from os import listdir

import time
# Face detector
face_detector = Face_Detector()

# Face identification
face_identification = Face_Identification()

names = ["dinesh", "erlic", "gilfoyle", "richard", "jared"]

images = [ cv2.imread("database/%s.png" % name) for name in names]

database = face_identification.generate_embeddings(images)

test_image = cv2.imread("test.jpg")

faces = face_detector.detect_image(test_image)

positions = face_detector.detect_positions(test_image)

start_time = time.time()

embeddings = face_identification.generate_embeddings(faces)

print("Faces: %s"%(time.time() - start_time))

for embedding in embeddings:

    distances = [pairwise_distance(embedding, emb) for emb in database]
    index = np.argmin(distances)
    
    print("name: %s:, distance: %f" %
          (names[index], pairwise_distance(embedding, database[index])))

    print("distances:", distances)



    







