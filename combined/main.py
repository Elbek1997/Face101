from library.face_detection import *
from library.face_identification import *

from os import listdir

import time
# Face detector
face_detector = Face_Detector()

# Face identification
face_identification = Face_Identification()

names = ["dinesh", "erlic", "gilfoyle", "richard"]

embeddings = [ face_identification.generate_embeddings(cv2.imread("database/%s.png" % name))  for name in names]




    







