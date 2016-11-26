import numpy as np
import cv2
import sys
import os.path

if len(sys.argv) != 3:
    print("wrong number of args")
    quit()

source_img = sys.argv[1]
target_img = sys.argv[2]

vertical_margin = 10
horizontal_margin = 10


if not os.path.isfile(source_img):
    print("wrong source file (name)")
    quit()

im = cv2.imread(source_img)
im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

ball_ycrcb_mint = np.array([20, 20, 20],np.uint8)
ball_ycrcb_maxt = np.array([255, 255, 255],np.uint8)
ball_ycrcb = cv2.inRange(im_ycrcb, ball_ycrcb_mint, ball_ycrcb_maxt)

areaArray = []
count = 1

contours, _ = cv2.findContours(ball_ycrcb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    areaArray.append(area)

sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

firstlargestcontour = sorteddata[0][1]

x, y, w, h = cv2.boundingRect(firstlargestcontour)

#cv2.drawContours(im, firstlargestcontour, -1, (255, 0, 0), 2)
#cv2.rectangle(im, (x, y), (x+w, y+h), (0,255,0), 2)
#cv2.imwrite('cds_flat2.jpg', im)

crop_img = im[y - vertical_margin: y + h + vertical_margin, x - horizontal_margin: x + w + horizontal_margin] # Crop from x, y, w, h -> 100, 200, 300, 400
# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
cv2.imwrite(target_img, crop_img)
