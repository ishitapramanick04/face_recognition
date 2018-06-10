import cv2
import numpy as np
import sys
import os
import math
import csv

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
##########################################################################################

def prepare_training_data(data_folder_path):
 dirs = os.listdir(data_folder_path)
 #print(dirs)
 #list to hold all subject faces
 faces = []
 #list to hold labels for all subjects
 labels = []
 kp=[]
 des=[]
 count=0
 #let's go through each directory and read images within it
 for dir_name in dirs:

    label = int(dir_name.replace("Subject", ""))
    subject_dir_path = data_folder_path + "/" + dir_name
    subject_images_names = os.listdir(subject_dir_path)

    #for each subfolder
    for image_name in subject_images_names:
        image_path = subject_dir_path + "/" + image_name

        #read image
        image = cv2.imread(image_path)
        count+=1

        #display an image window to show the image
        #cv2.imshow("Training on image...", image)
        #detect face
        face, rect = detect_face(image)
        #------STEP-4--------
        #for the purpose of this tutorial
        #we will ignore faces that are not detected
        if face is not None:
            #add face to list of faces
            #face=cv2.resize(face,(480, 640))
            print(face.shape," ",label)
            faces.append(face)
            #add label for this face
            labels.append(label)
            sift = cv2.xfeatures2d.SIFT_create()
            kp1,des1=sift.detectAndCompute(face,None)
            kp.append(kp1)
            des.append(des1)
        #cv2.destroyAllWindows()
 print("no of counts:",count)
 return kp,des,faces,labels

#############################################################################################
def test_data(data_folder_path,des,faces,kp,labels) :
    dirs = os.listdir(data_folder_path)
    with open('test.csv','a',newline='') as f:
         thewriter=csv.writer(f)
         for dir_name in dirs:
             test_dir_path = data_folder_path + "/" + dir_name
             image = cv2.imread(test_dir_path)
             face2, _ = detect_face(image)
             distance_min = math.inf
             #print(face.shape)
             #predict the image using our face recognizer
             if face2 is not None:
                 sift = cv2.xfeatures2d.SIFT_create()
                 kp2,des2=sift.detectAndCompute(face2,None)
                 bf = cv2.BFMatcher()
                 for des1,face1,kp1,label_this in zip(des,faces,kp,labels):
                     matches = bf.match(des1,des2)
                     matches = sorted(matches, key=lambda val: val.distance)
                     img = cv2.drawMatches(face1,kp1,face2,kp2,matches[0:20],None)
                     sum=0
                     if len(matches)>=10:
                        for i in range(10):
                            sum+=matches[i].distance
                            if sum < distance_min:
                                distance_min=sum
                                label_test=label_this
                                print("label is",label_test)
                     #cv2.imshow("Image",img)
                     #cv2.waitKey(100)
                 thewriter.writerow([test_dir_path,"SIFT",str(label_test),str(distance_min)])
                 #cv2.destroyAllWindows()
#############################################################################################
kp,des,faces,labels=prepare_training_data("training_dataset")
test_data("test_data",des,faces,kp,labels)
exit()
