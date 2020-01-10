import cv2
import os
import matplotlib.pyplot as plt
import spasavanjeSlika as spasiSliku

def increaseContrast (image, factor):
    im2 = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    mean = cv2.mean(im2)
    height, width, channels = im2.shape
    for x in range(0, height):
        for y in range(0, width):
            if im2[x,y,1] < mean[1] and im2[x,y,1] > factor:
                im2[x,y][1] -= factor
            if im2[x,y,1] > mean[1] and im2[x,y,1] < 255 - factor:
                im2[x,y,1] += factor
    return cv2.cvtColor(im2, cv2.COLOR_HLS2BGR)



slike_folder = os.listdir("images/milorad_dodik")

i = 0
for slike in slike_folder:
    image = cv2.imread("images/milorad_dodik/" + slike)
    enhancedImage = increaseContrast(image, 10)
    spasiSliku.spasiSliku("Contrast", "image", i, enhancedImage)
    i += 1

slike_folder = os.listdir("images/bakir_izetbegovic")
for slike in slike_folder:
    image = cv2.imread("images/bakir_izetbegovic/" + slike)
    enhancedImage = increaseContrast(image, 10)
    spasiSliku.spasiSliku("Contrast", "image", i, enhancedImage)
    i += 1   

slike_folder = os.listdir("images/dragan_covic")
for slike in slike_folder:
    image = cv2.imread("images/dragan_covic/" + slike)
    enhancedImage = increaseContrast(image, 10)
    spasiSliku.spasiSliku("Contrast", "image", i, enhancedImage)
    i += 1     
