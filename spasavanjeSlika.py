import cv2
import os


def spasiSliku(folder, name, number, picture):

	if (os.path.isdir(folder)==False):
		os.mkdir(folder)
	try:
		cv2.imwrite('./{}/{}{}.jpg'.format(folder, name, number), picture)
	except cv2.error as e:
		print('Greska')