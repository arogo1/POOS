import cv2
import os
import spasavanjeSlika as spasiSliku
from matplotlib import pyplot as plt
##from PIL import Image, ImageEnhance


def histogram(image, i, slika):

    img = image
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    spasiSliku.spasiSliku("Histogram", slika, i, final)    


slike_folder = os.listdir("images/milorad_dodik")

i = 0
for slike in slike_folder:
    image = cv2.imread("images/milorad_dodik/" + slike)
    histogram(image,i, "dodik")
    i += 1

slike_folder = os.listdir("images/bakir_izetbegovic")
i = 0
for slike in slike_folder:
    image = cv2.imread("images/bakir_izetbegovic/" + slike)
    histogram(image,i, "bakir")
    i += 1    

slike_folder = os.listdir("images/dragan_covic")
i = 0
for slike in slike_folder:
    image = cv2.imread("images/dragan_covic/" + slike)
    histogram(image,i, "covic")
    i += 1       
