import dlib
import cv2
import os
import glob
from skimage import io
import imutils
from imutils import face_utils
import cv2
import os
import numpy as np
from imutils.face_utils.facealigner import FaceAligner
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa=FaceAligner(sp)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread('cr71.png')
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# show the original input image and detect faces in the grayscale
# image
cv2.imshow("Input", image)
detector = dlib.get_frontal_face_detector()
rects = detector(gray, 2)
for rect in rects:
	# extract the ROI of the *original* face, then align the face
	# using facial landmarks
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
	faceAligned = fa.align(image, gray, rect)

	# display the output images
	cv2.imshow("Original", faceOrig)
	cv2.imshow("Aligned", faceAligned)
	cv2.waitKey(0)
