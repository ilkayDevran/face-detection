import dlib
import cv2
import time
from matplotlib import pyplot as plt


start_time = time.time()
path = 'test_images/image20.jpg'
img = cv2.imread(path ,cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dnnFaceDetector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

rects = dnnFaceDetector(gray, 1)
for (i, rect) in enumerate(rects):
    x1 = rect.rect.left()
    y1 = rect.rect.top()
    x2 = rect.rect.right()
    y2 = rect.rect.bottom()
    # Rectangle around the face
    cv2.rectangle(gray, (x1, y1), (x2, y2), (255, 255, 255), 3)

print("--- %s seconds ---" % (time.time() - start_time))
plt.figure(figsize=(12,8))
plt.imshow(gray, cmap='gray')
plt.show()