#import libraries
import cv2
import numpy as np
import os


class faceChopper:
	def __init__(self):
		# Import Classifier for Face and Eye Detection
		self.frontal_face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		self.profile_face_detector = cv2.CascadeClassifier('haarcascade_profileface.xml')

	def face_detector(self, img, size=0.5):
		try:
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		except Exception as e:
			return 0,img

		faces = self.frontal_face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
		if faces is ():
			faces = self.profile_face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
			if faces is ():
				return 0, img
		
		# Given coordinates to detect face and eyes location from ROI
		print(len(faces))
		roi_color=[]
		for (x, y, w, h) in faces:
			c = 15
			x = x - c 
			w = w + c 
			y = y - c 
			h = h + c
			roi_gray = gray[y: y+h, x: x+w]
			roi_color.append( img[y: y+h, x: x+w])

		print(roi_color.__len__())
		return roi_color.__len__(), roi_color


def main():
	fC = faceChopper()

	for imageName in os.listdir("inputs"):
		if imageName != '.DS_Store':
			face_counter=0
			print(imageName)
			cap = cv2.imread("inputs/"+imageName)
			#cap = cv2.resize(cap, (400, 300)) 
			_,frames = fC.face_detector (cap)
			for frame in frames:
				cv2.imwrite(os.path.join("outputs", imageName.replace('.jpg','') + "-face%d.jpg" % face_counter), frame)
				face_counter+=1


if __name__ == "__main__":
	main()