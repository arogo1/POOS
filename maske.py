import cv2
import numpy as np
import json
import os
import spasavanjeSlika as ss

def maske():

    f = open('anotacije/data.json')  
    lines = f.readlines()
    data = []
    for line in lines:
        d = json.loads(line)
        data.append(d)
    c = 0
    for d in data:
        annotations = d['people']

        for annotation in annotations:
            name = str(annotation['name'])
            h = int(annotation['h'])
            w = int(annotation['w'])
            x = int(annotation['x'])
            y = int(annotation['y'])
            location = "images/"
            if "bakir" not in name:
                if "dodik" not in name:
                    location += "dragan_covic/"
                else:
                    location += "milorad_dodik/"
            else:
                location += "bakir_izetbegovic/"
            img = cv2.imread(location + name + ".jpg" ,0)
            print(location + name)
            if img is not None:
                mask = np.zeros(img.shape, dtype="uint8")
                cv2.rectangle(mask, (x,y), (x+w, y+h), (255, 255, 255), -1)
                maskedImg = cv2.bitwise_and(img, mask)
                ss.spasiSliku("maske", name, 1, maskedImg)
maske()
