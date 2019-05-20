import os, cv2
import numpy as np
import matplotlib.pyplot as plt

from_dir = 'train/lego_f_train_image'
to_dir = 'train_gray/lego_f_train_image'
for image in os.listdir(from_dir):
    origin = cv2.imread(from_dir + '/' + image)
    gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(to_dir + '/' + image, gray)
