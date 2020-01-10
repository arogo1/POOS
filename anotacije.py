import cv2
import sys
import numpy
import json
import spasavanjeSlika as ss

# Get user supplied values
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

folder="images/bakir_izetbegovic"
slika="bakir"

data = {}  
data['people'] = []  

# Oznacavanje anotacija Bakira
for i in range(1,21):
    image = cv2.imread("{}/{}{}.jpg".format(folder, slika, i))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))
    a,b,c,d = 0,0,0,0
    for (x,y,w,h) in faces:
         if(c < w and d < h):
            a = x
            b = y
            c = w
            d = h
    # Draw a rectangle around the faces
    cv2.rectangle(image, (a, b), (a+c, b+d), (0, 255, 0), 2)
    ss.spasiSliku('anotacije', "bakir", i,image)
    data['people'].append({  
        'name': 'bakir' + str(i),
        'x': str(a),
        'y': str(b),
        'w': str(c),
        'h': str(d)
     })

folder="images/milorad_dodik"
slika="dodik"


# Oznacavanje anotacija Dodika
for i in range(1,21):
    image = cv2.imread("{}/{}{}.jpg".format(folder, slika, i))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
    ss.spasiSliku('anotacije', "dodik", i,image)
    data['people'].append({  
        'name': 'dodik' + str(i),
        'x': str(a),
        'y': str(b),
        'w': str(c),
        'h': str(d)
     })

folder="images/dragan_covic"
slika="covic"


# Oznacavanje anotacija Covic
for i in range(1,21):
    image = cv2.imread("{}/{}{}.jpg".format(folder, slika, i))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
    ss.spasiSliku('anotacije', "covic", i,image)
    data['people'].append({  
        'name': 'covic' + str(i),
        'x': str(a),
        'y': str(b),
        'w': str(c),
        'h': str(d)
     })

with open('anotacije/data.json', 'w') as outfile:  
    json.dump(data, outfile)
