import cv2
import numpy as np
import sys
def detect_face(img):
    #convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('/home/ishita/opencv/data/lbpcascades/lbpcascade_frontalface_improved.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    #if no faces are detected then return original img
    if (len(faces) == 0):
        face_cascade = cv2.CascadeClassifier('/home/ishita/opencv/data/lbpcascades/lbpcascade_frontalface.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
        if (len(faces) == 0):
            face_cascade = cv2.CascadeClassifier('/home/ishita/opencv/data/lbpcascades/lbpcascade_profileface.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
            if (len(faces) == 0):
                return None, None
        x, y, w, h = faces[0]
        return gray[y:y+w, x:x+h], faces[0]
#return only the face part of the image
    x, y, w, h = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

img1=cv2.imread('1.jpg')
img2=cv2.imread('2.jpg')
face1,_=detect_face(img1)
face2,_=detect_face(img2)
if len(face1) == 0 :
    print("1")
    exit()
elif len(face2) == 0 :
    print2("2")
    exit()
#cv2.imshow("Image",face1)
#cv2.waitKey(0)
#cv2.imshow("Image",face2)
#cv2.waitKey(0)
sift = cv2.xfeatures2d.SIFT_create()
kp1,des1=sift.detectAndCompute(face1,None)
kp2,des2=sift.detectAndCompute(face2,None)
bf = cv2.BFMatcher()
matches = bf.match(des1,des2)
matches = sorted(matches, key=lambda val: val.distance)
img = cv2.drawMatches(face1,kp1,face2,kp2,matches,None)
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()
