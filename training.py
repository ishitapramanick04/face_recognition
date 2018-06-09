import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import csv
subjects = ["none", "1", "2","3","4","5","6"]
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


def prepare_training_data(data_folder_path):
 dirs = os.listdir(data_folder_path)
 #print(dirs)
 #list to hold all subject faces
 faces = []
 #list to hold labels for all subjects
 labels = []
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
        #cv2.waitKey(100)

        #detect face
        face, rect = detect_face(image)
        #------STEP-4--------
        #for the purpose of this tutorial
        #we will ignore faces that are not detected
        if face is not None:
            #add face to list of faces
            face=cv2.resize(face,(480, 640))
            print(face.shape)
            faces.append(face)
            #add label for this face
            labels.append(label)

        #cv2.destroyAllWindows()
 #cv2.waitKey(1)
 cv2.destroyAllWindows()
 print("no of counts:")
 print(count)
 return faces, labels



#image recogniser
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#function to draw text on give image starting from
#passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

#test data_folder_path
def test_data(data_folder_path) :
    dirs = os.listdir(data_folder_path)
    with open('test.csv','a',newline='') as f:
         thewriter=csv.writer(f)
         for dir_name in dirs:
             test_dir_path = data_folder_path + "/" + dir_name
             image = cv2.imread(test_dir_path)
             face, rect = detect_face(image)
             #print(face.shape)
             #predict the image using our face recognizer
             if face is not None:
                 face=cv2.resize(face,(480, 640))
                 label,confidence= face_recognizer.predict(face)
                 label_text = str(subjects[label])
                 confidence_text= str(confidence)
                 #draw a rectangle around face detected

                 thewriter.writerow(["fisher",test_dir_path,label_text,confidence])
                 draw_rectangle(image, rect)
                 #draw name of predicted person
                 draw_text(image, label_text+' '+confidence_text, rect[0], rect[1]-5)
                 #cv2.imshow("The prediction", predictx)
                 #cv2.waitKey(0)

print("Preparing data...")
faces, labels = prepare_training_data("training_dataset")
print("Data prepared")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

#face_recognizer=cv2.face_LBPHFaceRecognizer.create()
#face_recognizer =cv2.face_EigenFaceRecognizer.create()
face_recognizer=cv2.face_FisherFaceRecognizer.create()
face_recognizer.train(faces, np.array(labels))
test_data('test_data')
exit()
