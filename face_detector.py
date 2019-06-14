import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob



txtfiles = [] 
for file in glob.glob("*.jpg"):
    txtfiles.append(file)
    
for ix in txtfiles:
    img = cv2.imread(ix,cv2.IMREAD_COLOR)
    imgtest1 = img.copy()
    imgtest = cv2.cvtColor(imgtest1, cv2.COLOR_BGR2GRAY)
    facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
   
    faces = facecascade.detectMultiScale(imgtest, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    print('Total number of Faces found',len(faces))
    for (x, y, w, h) in faces:
        face_detect = cv2.rectangle(imgtest, (x, y), (x+w, y+h), (0, 0, 0), 2)
        roi_gray = imgtest[y:y+h, x:x+w]
        roi_color = imgtest[y:y+h, x:x+w]        
        plt.imshow(face_detect)
    #plt.show()

    """
    eyes = eye_cascade.detectMultiScale(roi_gray)
    print('Total number of Eyes found',len(eyes))
    for (ex,ey,ew,eh) in eyes:
        eye_detect = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,0),2)
        plt.imshow(eye_detect)
    #plt.show()
    """

    profile_faces = profile_cascade.detectMultiScale(imgtest, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    print('Total number of Profile Faces found',len(profile_faces))
    for (x, y, w, h) in profile_faces:
        profile_face_detect = cv2.rectangle(imgtest, (x, y), (x+w, y+h), (255, 0, 255), 2)
        roi_gray = imgtest[y:y+h, x:x+w]
        roi_color = imgtest[y:y+h, x:x+w]        
        plt.imshow(profile_face_detect)
    plt.show()

print("Done")