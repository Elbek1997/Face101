from library.face_detection import *
from library.face_identification import *

import time
# Face detector
face_detector = Face_Detector()

# Face identification
face_identification = Face_Identification()


image = cv2.imread("test.jpg")

faces = face_detector.detect_image(image)

# for face in faces:
    
#     embeddings = face_identification.generate_embeddings(face)
#     print(face.shape, embeddings)

face_identification.generate_embeddings(faces[-1])

start_time = time.time()
face_identification.generate_embeddings(faces[0])
print("%s seconds\n" % (time.time() - start_time) )


