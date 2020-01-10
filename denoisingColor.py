import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

import spasavanjeSlika as spasiSliku

def denoisingColor(image, i):
    img = image

    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)


    spasiSliku.spasiSliku("colorDenoising", "slika", i, dst)


slike_folder = os.listdir("images/barack_obama")

i = 0
for slike in slike_folder:
    image = cv2.imread("images/barack_obama/" + slike)
    denoisingColor(image,i)
    i += 1

slike_folder = os.listdir("images/mark_zuckerberg")
for slike in slike_folder:
    image = cv2.imread("images/mark_zuckerberg/" + slike)
    denoisingColor(image,i)
    i += 1    

slike_folder = os.listdir("images/jimmy_kimmel")
for slike in slike_folder:
    image = cv2.imread("images/jimmy_kimmel/" + slike)
    denoisingColor(image,i)
    i += 1       
