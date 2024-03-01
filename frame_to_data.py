# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 22:46:45 2024

@author: User
"""

import cv2
import numpy as np
import mss
from PIL import ImageGrab
import win32api
import win32con
import ait
import time 

path = 'cascade.xml'
scale = 2 # lower scale = better detection = lower performance
min_neig = 4 # lower minimum neighbour = more detection (might be false detection)
cascade = cv2.CascadeClassifier(path)

'''image detaching

vidcap = cv2.VideoCapture('data/Human Video/HV1.mp4')
success,image = vidcap.read()
count = 0
success = True

while success:
  success,image = vidcap.read()
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1
'''
l=[]
count =0
n= 1945#frame amount
for i in range(n):
    img = cv2.imread(f'data/AI data/AV1 frame/frame{count}.jpg')


    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    detection = cascade.detectMultiScale(gray, scale, min_neig)
    found=len(detection)
    count +=1
    if found!=0:
        apx = (detection[0][0] + (detection[0][2] / 2)) - 1600/2
        apy = (detection[0][1] + (detection[0][3] / 2)) - 900/2
        l.append((round(apx),round(apy)))
        print(apx,apy,count)
data=np.array(l)
print(data.shape)
np.save('data/AI data/AI-data 1.npy',data)    


#frame to data    