from library.face_detection import *
from library.face_identification import *
from library.utils import *

from os import listdir
from os.path import splitext, join, isfile

from argparse import ArgumentParser

def main(args):
      # Face detector
      face_detector = Face_Detector()

      # Face identification
      face_identification = Face_Identification()

      # Database names
      names = []

      # Images
      images = []
      
      # Embeddings
      database = []

      # Load database
      for file in listdir(args.database):
            
            if file.endswith( ('.jpg', '.png', '.bmp') ) and isfile(join(args.database, file)):

                  names.append(splitext(file)[0])
                  images.append(cv2.imread(join(args.database, file)))

      
      database = face_identification.generate_embeddings(images)

      test_image = cv2.imread(args.input)

      # Detect face positions
      positions = face_detector.detect_positions(test_image)

      # Generate embeddings
      embeddings = face_identification.generate_embeddings([ test_image[startY:endY, startX:endX] for (startX, startY, endX, endY) in positions])


      for position, embedding in zip(positions, embeddings):

            # Draw rectangle
            test_image = drawRectangle(test_image, position)
            
            # Generate distances
            distances = [ pairwise_distance(emb, embedding) for emb in database]
            
            # Find min index
            index = np.argmin(distances)

            print(index, distances)

            txt = "%s_%.2f" % (names[index], 1 - distances[index])

            # Draw name
            test_image = drawString(test_image, txt, (position[0], position[1]-10))
            
      
      cv2.imwrite(args.output, test_image)



parser = ArgumentParser()

parser.add_argument("--database", type=str, default="database", help="Path to database")

parser.add_argument("--input", type=str, default="./test.jpg", help="Path to input image")

parser.add_argument("--output", type=str, default="./out.jpg", help="Path to output image")

args = parser.parse_args()

main(args)




    







