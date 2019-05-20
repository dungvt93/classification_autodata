import cv2
import numpy as np
from matplotlib import pyplot as plt



# img = cv2.imread('test.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# kernel_size = 5
# blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
# edges = cv2.Canny(img,100,120)
#
# lines = cv2.HoughLinesP(edges,1,np.pi/180,40,minLineLength=30,maxLineGap=30)
# print(lines.shape)
#
# i = 0
# for line in lines:
#     for x1,y1,x2,y2 in line:
#         i+=1
#         cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
# cv2.imshow("res",img)
# cv2.waitKey(0)

# 輪郭を抽出する。
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# print(contours)
# cv2.drawContours(gray, contours, -1, (0, 255, 0), 20)
# cv2.imshow('contours', gray)
#
# cv2.waitKey()

from PIL import Image
import numpy as np
import os,random
for i in range(1,30):
    img = Image.open("origin/" + random.choice(os.listdir("origin")), 'r').convert('RGBA')
    img = img.resize((232,105))
    img = img.rotate(-3,expand=1)
    img_w, img_h = img.size

    background = Image.open("background/" + random.choice(os.listdir("background")),'r').convert('RGBA')
    background = background.resize((256,256))
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) // 2 + np.random.randint(low=-12,high=12), (bg_h - img_h) // 2 + np.random.randint(low=-75,high=75) )
    background.paste(img, offset, img)
    background.save("result/%05d.png" % i)
    print("%05d.png" % i)



