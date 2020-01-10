import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import spasavanjeSlika as spasiSliku


def sharpenimage ():
    slike_folder = os.listdir("colorDenoising")

    i = 0
    for slike in slike_folder:
        slike = cv2.imread("colorDenoising/slika" + str(i) + ".jpg")
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(slike, -1, kernel)
        spasiSliku.spasiSliku("SharpenedImages", "slika", i, sharpened)
        i += 1


sharpenimage()   
