import cv2
import numpy as np
import mss
from PIL import ImageGrab
import win32api
import win32con
import ait
import time 

path = 'model/cascade.xml' # cascade path
obj = 'target' # detected obj name (for display)

# cascade values for tweaking
scale = 2       # lower scale = better detection = lower performance
min_neig = 4    # lower minimum neighbour = more detection (might be false detection)

WIDTH = 1600
HEIGHT = 900

cascade = cv2.CascadeClassifier(path)

# Record video setting 
image = ImageGrab.grab()
width, height = image.size
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('test.avi', fourcc, 30, (WIDTH, HEIGHT))

l = []

while True:
    while win32api.GetKeyState(0x26)<0:
        #start = time.time()
        time.sleep(0.005) # wait for the ball to come out 
        with mss.mss() as sct:
            monitor = {"top": 0, "left": 0, "width": WIDTH, "height": HEIGHT}
            img_array = np.array(sct.grab(monitor))
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        detection = cascade.detectMultiScale(gray, scale, min_neig)
        if len(detection) != 0:
            apx = (detection[0][0] + (detection[0][2] / 2)) - WIDTH / 2
            apy = (detection[0][1] + (detection[0][3] / 2)) - HEIGHT / 2
        
            n = 1
            for i in range(n):
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, round(apx), 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 0, round((apy)))
                # l.append((round((apx)*0.89/n),round((apy)*0.89/n)))
                # img_rgb = ImageGrab.grab()
                # img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
                # video.write(img_bgr) #grab frame to recrod video
     
        
        ait.click()
        
        
        # end = time.time()
        # print(end - start)
        
        a = win32api.GetKeyState(0x28)  # down arrow key
        if a < 0:
            break