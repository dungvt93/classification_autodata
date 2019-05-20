from scipy import ndimage
import cv2
import datetime
import os

def rotate(img, degree):
	return ndimage.rotate(img, degree)



def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def increase_brightness_rate(img, rate):
	i=0
	while(i<rate):
		i = i + 5 
		cv2.imwrite(str(datetime.datetime.now()) + str(i)+'.jpg',increase_brightness(img,i))
		print(str(datetime.datetime.now()) + str(i) + '.jpg')

def rotate_rate(img,rate):
	i=-rate
	while (i<rate):
		i = i + 1
		cv2.imwrite(str(datetime.datetime.now()) + str(i)+'.jpg',rotate(img,i))
		print(str(i) + '.jpg')


img = cv2.imread('test2.png')
dir_name = 'train/lego_anomaly'
for file_name in os.listdir(dir_name):
	img = cv2.imread(dir_name + "/" + file_name)
	increase_brightness_rate(img,30)
	#rotate_rate(img,5)

