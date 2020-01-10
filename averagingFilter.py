import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import spasavanjeSlika as spasiSliku



def averagingFilter(image,i,slika):
    img = image.copy()
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(img,-1,kernel)

    spasiSliku.spasiSliku('averagingFilter', slika, i, dst)

slike_folder = os.listdir("images/milorad_dodik")

i = 0
for slike in slike_folder:
    image = cv2.imread("images/milorad_dodik/" + slike)
    averagingFilter(image,i, "dodik")
    i += 1

slike_folder = os.listdir("images/bakir_izetbegovic")
i = 0
for slike in slike_folder:
    image = cv2.imread("images/bakir_izetbegovic/" + slike)
    averagingFilter(image,i, "bakir")
    i += 1    

slike_folder = os.listdir("images/dragan_covic")
i = 0
for slike in slike_folder:
    image = cv2.imread("images/dragan_covic/" + slike)
    averagingFilter(image,i, "covic")
    i += 1       
