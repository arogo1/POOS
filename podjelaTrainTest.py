import random as random
import os
import cv2

folder="images/milorad_dodik"
slika="dodik"

if (os.path.isdir("Train") == False):
	os.mkdir("Train")

if (os.path.isdir("Test") == False):
	os.mkdir("Test")

#TestDodik
for i in range(1,10):
	I = cv2.imread("{}/{}{}.jpg".format(folder, slika, i))
	cv2.imwrite('Test/{}{}.jpg'.format(slika, i), I)


#TrainDodik
for i in range(10,18):
	I = cv2.imread("{}/{}{}.jpg".format(folder, slika, i))
	cv2.imwrite('Train/dodik/{}{}.jpg'.format(slika, i), I)

folder="images/dragan_covic"
slika="covic"

#TestCovic
for i in range(1,10):
	I = cv2.imread("{}/{}{}.jpg".format(folder, slika, i))
	cv2.imwrite('Test/{}{}.jpg'.format(slika, i), I)


#TrainCovic
for i in range(10,18):
	I = cv2.imread("{}/{}{}.jpg".format(folder, slika, i))
	cv2.imwrite('Train/covic/{}{}.jpg'.format(slika, i), I)



folder="images/bakir_izetbegovic"
slika="bakir"

#TestBakir
for i in range(1,10):
	I = cv2.imread("{}/{}{}.jpg".format(folder, slika, i))
	cv2.imwrite('Test/{}{}.jpg'.format(slika, i), I)


#TrainBakir
for i in range(10,18):
	I = cv2.imread("{}/{}{}.jpg".format(folder, slika, i))
	cv2.imwrite('Train/bakir/{}{}.jpg'.format(slika, i), I)
