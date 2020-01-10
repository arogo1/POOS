import cv2



def face_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    a,b,c,d = 0,0,0,0
    i = 0
    indeks = 0
    for (x,y,w,h) in faces:
         if(c < w and d < h):
            a = x
            b = y
            c = w
            d = h
            indeks = i
         i += 1
       
    return gray[b:b + c, a:a + d], faces[indeks]
