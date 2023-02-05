import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector


HEIGHT, WIDTH = 480, 640


cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (WIDTH, HEIGHT))
    img, faces = detector.findFaceMesh(img)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
