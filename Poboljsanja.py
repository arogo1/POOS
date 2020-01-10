import cv2
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import faceDetector as fd

poznata_lica = ["Kimmel", "Mark", "Obama"]
predicted_subjects = []
face_recognizer = cv2.face.LBPHFaceRecognizer_create()


def train_data(data_folder_path):
    faces = []
    names = []
    name = 0
    slike_folder = os.listdir("Train1/kimmel")
    for slike in slike_folder:
            print(slike)
            image = cv2.imread("Train1/kimmel/" + slike)
            face, rect = fd.face_detector(image)
            faces.append(face)
            names.append(name)
    name = 1
    slike_folder = os.listdir("Train1/mark")
    for slike in slike_folder:
            print(slike)
            image = cv2.imread("Train1/mark/" + slike)
            face, rect = fd.face_detector(image)
            faces.append(face)
            names.append(name)
    name = 2
    slike_folder = os.listdir("Train1/obama")
    for slike in slike_folder:
            print(slike)
            image = cv2.imread("Train1/obama/" + slike)
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

def main():

        faces, names = train_data("Train1")

        face_recognizer = cv2.face.LBPHFaceRecognizer_create()


        face_recognizer.train(faces, np.array(names))
        face_recognizer.save('train1.yml')


        slike_folder = os.listdir("Test1/")
        slike = []
        brojac = 0
        real_subjects = []
        mark_real = []
        mark_prediction = []

        for slika in slike_folder:
            image = cv2.imread("Test1/" + slika)
            if str(slika).startswith('mark'):
                    real_subjects.append("Mark")
            if str(slika).startswith('kimmel'):
                    real_subjects.append("Kimmel")
            if str(slika).startswith('obama'):
                    real_subjects.append("Obama")
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

        TP,FP,TN,FN = perf_measure(real_subjects, predicted_subjects, "Mark")

        print("Mark: ")
        print("acc: " + str(float(TP + TN)/float(TP + FP + TN + FN)))
        print("spec: " + str(float(TN)/float(TN + FP)))
        print("sens: " + str(float(TP)/float(TP + FN)))

        TP,FP,TN,FN = perf_measure(real_subjects, predicted_subjects, "Kimmel")

        print("Kimmel: ")
        print("acc: " + str(float(TP + TN)/float(TP + FP + TN + FN)))
        print("spec: " + str(float(TN)/float(TN + FP)))
        print("sens: " + str(float(TP)/float(TP + FN)))


        TP,FP,TN,FN = perf_measure(real_subjects, predicted_subjects, "Obama")

        print("Obama: ")
        print("acc: " + str(float(TP + TN)/float(TP + FP + TN + FN)))
        print("spec: " + str(float(TN)/float(TN + FP)))
        print("sens: " + str(float(TP)/float(TP + FN)))


main()





