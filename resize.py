import os
import cv2
from_dir = "train"
for file_name in  os.listdir(from_dir):
	img = cv2.imread(from_dir + "/" + file_name)
	img = cv2.resize(img,(256,256))
	cv2.imwrite("result/"+file_name, img)
