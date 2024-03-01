import cv2
import numpy as np
import mss
from PIL import ImageGrab
import win32api
import win32con
import ait
import time 

vidcap = cv2.VideoCapture('data/AI video/AV1.mp4')
success,image = vidcap.read()
count = 0
success = True

while success:
  success,image = vidcap.read()
  cv2.imwrite("data/AI data/AV1 frame/frame%d.jpg" % count, image)     # save frame as JPEG file
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1

#Video to frame