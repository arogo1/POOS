import cv2
import os
import json
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import spasavanjeSlika as ss
import faceDetector as fd

poznata_lica = ["Covic", "Bakir", "Milorad"]
predicted_subjects = []


def train_data(data_folder_path):
    faces = []
    names = []
    name = 0
    slike_folder = os.listdir("Train/covic")
    for slike in slike_folder:
            print(slike)
            image = cv2.imread("Train/covic/" + slike)
            face, rect = fd.face_detector(image)
            faces.append(face)
            names.append(name)
    name = 1
    slike_folder = os.listdir("Train/bakir")
    for slike in slike_folder:
            print(slike)
            image = cv2.imread("Train/bakir/" + slike)
            face, rect = fd.face_detector(image)
            faces.append(face)
            names.append(name)
    name = 2
    slike_folder = os.listdir("Train/dodik")
    for slike in slike_folder:
            print(slike)
            image = cv2.imread("Train/dodik/" + slike)
            face, rect = fd.face_detector(image)
            faces.append(face)
            names.append(name)

    return faces, names




def perf_measure(y_actual, y_pred, person):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_actual[i]==y_pred[i]==person:
           TP += 1
        if y_pred[i]== person and y_actual[i]!=y_pred[i]:
           FP += 1
        if y_actual[i]==y_pred[i]!=person:
           TN += 1
        if y_pred[i]!=person and y_actual[i]!=y_pred[i]:
           FN += 1

    return(TP, FP, TN, FN)


def perf_measure(y_actual, y_pred, person):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_actual[i]==y_pred[i]==person:
           TP += 1
        if y_pred[i]== person and y_actual[i]!=y_pred[i]:
           FP += 1
        if y_actual[i]==y_pred[i]!=person:
           TN += 1
        if y_pred[i]!=person and y_actual[i]!=y_pred[i]:
           FN += 1

    return(TP, FP, TN, FN)

data = {}  
data['people'] = []
def deskriptor(imag,i):
    image = imag
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))
    # Draw a rectangle around the faces
    a,b,c,d = 0,0,0,0
    for (x,y,w,h) in faces:
         if(c < w and d < h):
            a = x
            b = y
            c = w
            d = h
    # Draw a rectangle around the faces
    cv2.rectangle(image, (a, b), (a+c, b+d), (0, 255, 0), 2)
    ss.spasiSliku('deskriptor', "slika", i,image)
    data['people'].append({  
        'name': 'slika' + str(i),
        'x': str(a),
        'y': str(b),
        'w': str(c),
        'h': str(d)
     })

    with open('deskriptor/data.json', 'w') as outfile:  
         json.dump(data, outfile)    

def main(path):

        faces, names = train_data("Train")

        face_recognizer = cv2.face.LBPHFaceRecognizer_create()  


        face_recognizer.train(faces, np.array(names))
        face_recognizer.save('train.yml')


        slike_folder = os.listdir(path)
        slike = []
        brojac = 0
        real_subjects = []
        mark_real = []
        mark_prediction = []
        i = 0 
        for slika in slike_folder:
                image = cv2.imread(path + "/" + slika)
                deskriptor(image,i)
                i += 1
                if str(slika).startswith('bakir'):
                        real_subjects.append("Bakir")
                if str(slika).startswith('covic'):
                        real_subjects.append("Covic")
                if str(slika).startswith('dodik'):
                        real_subjects.append("Dodik")
                predicted_image = image.copy()
                face, face1 = fd.face_detector(predicted_image)
                label, confidence = face_recognizer.predict(face)
                text = poznata_lica[label]
                cv2.rectangle(predicted_image, (face1[0], face1[1]), (face1[0] + face1[2], face1[1] + face1[3]), (0, 255, 0), 2)
                cv2.putText(predicted_image, text, (face1[0], face1[1] - 2), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                predicted_subjects.append(text)
                slike.append(predicted_image)
                cv2.imshow("Treniranje", cv2.resize(predicted_image, (550, 600)))
                cv2.waitKey(1500)

       
        print(real_subjects)
        print(predicted_subjects)

        print("accuracy: " + str(accuracy_score(real_subjects,predicted_subjects)))

        TP,FP,TN,FN = perf_measure(real_subjects, predicted_subjects, "Bakir")

        print("Bakir: ")
        print("acc: " + str(float(TP + TN)/float(TP + FP + TN + FN)))
        print("spec: " + str(float(TN)/float(TN + FP)))
        print("sens: " + str(float(TP)/float(TP + FN)))

        TP,FP,TN,FN = perf_measure(real_subjects, predicted_subjects, "Covic")

        print("Covic: ")
        print("acc: " + str(float(TP + TN)/float(TP + FP + TN + FN)))
        print("spec: " + str(float(TN)/float(TN + FP)))
        print("sens: " + str(float(TP)/float(TP + FN)))


        TP,FP,TN,FN = perf_measure(real_subjects, predicted_subjects, "Dodik")

        print("Dodik: ")
        print("acc: " + str(float(TP + TN)/float(TP + FP + TN + FN)))
        print("spec: " + str(float(TN)/float(TN + FP)))
        print("sens: " + str(float(TP)/float(TP + FN)))


main("images/posebanFolder")


