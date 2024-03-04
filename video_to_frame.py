import cv2

vidcap = cv2.VideoCapture('data/AI video/AV1.mp4')
success,image = vidcap.read()
count = 0
success = True

while success:
    success,image = vidcap.read()
    cv2.imwrite("data/AI data/AV1 frame/frame%d.jpg" % count, image)     # save frame as JPEG file
    if cv2.waitKey(10) == 27:                                            # exit if Escape is hit
        break
    count += 1
