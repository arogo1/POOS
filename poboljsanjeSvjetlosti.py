import cv2
import os
import matplotlib.pyplot as plt
import spasavanjeSlika as spasiSliku

def increaseBrightness (image, factor):
    im2 = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    height, width, channels = im2.shape
    for x in range(0, height):
        for y in range(0, width):
            if im2[x,y,1] < 255 - factor:
                im2[x,y,1] += factor
    return cv2.cvtColor(im2, cv2.COLOR_HLS2BGR)


slike_folder = os.listdir("images/milorad_dodik")

i = 0
for slike in slike_folder:
    image = cv2.imread("images/milorad_dodik/" + slike)
    enhancedImage = increaseBrightness(image, 15)
    spasiSliku.spasiSliku("Brightness", "image", i, enhancedImage)
    i += 1

slike_folder = os.listdir("images/bakir_izetbegovic")
for slike in slike_folder:
    image = cv2.imread("images/bakir_izetbegovic/" + slike)
    enhancedImage = increaseBrightness(image, 15)
    spasiSliku.spasiSliku("Brightness", "image", i, enhancedImage)
    i += 1   

slike_folder = os.listdir("images/dragan_covic")
for slike in slike_folder:
    image = cv2.imread("images/dragan_covic/" + slike)
    enhancedImage = increaseBrightness(image, 15)
    spasiSliku.spasiSliku("Brightness", "image", i, enhancedImage)
    i += 1     
