import numpy as np
import cv2
import sys
import os
import pickle

directory = './cds_pics'

samples = []

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        print(os.path.join(directory, filename))

        source_img = os.path.join(directory, filename)
        target_img = os.path.join(directory, filename)[:-4] + '_2' + '.jpg'

        vertical_margin = 10
        horizontal_margin = 10


        if not os.path.isfile(source_img):
            print("wrong source file (name)")
            quit()

        im = cv2.imread(source_img)
        im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        ball_ycrcb_mint = np.array([0, 0, 0],np.uint8)
        ball_ycrcb_maxt = np.array([250, 250, 250],np.uint8)
        ball_ycrcb = cv2.inRange(im_ycrcb, ball_ycrcb_mint, ball_ycrcb_maxt)

        areaArray = []
        count = 1

        contours, _ = cv2.findContours(ball_ycrcb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            areaArray.append(area)

        sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

        contour1 = sorteddata[0][1]
        contour2 = sorteddata[1][1]
        contour3 = sorteddata[2][1]

        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        cv2.drawContours(im, contour1, -1, (255, 0, 0), 2)
        cv2.rectangle(im, (x1, y1), (x1+w1, y1+h1), (255,50,50), 2)

        '''crop_img = im[y - vertical_margin: y + h + vertical_margin, x - horizontal_margin: x + w + horizontal_margin] # Crop from x, y, w, h -> 100, 200, 300, 400
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        cv2.imwrite(target_img, crop_img)'''

        x2, y2, w2, h2 = cv2.boundingRect(contour2)
        cv2.drawContours(im, contour2, -1, (0, 255, 0), 2)
        cv2.rectangle(im, (x2, y2), (x2+w2, y2+h2), (50,255,50), 2)

        x3, y3, w3, h3 = cv2.boundingRect(contour3)
        cv2.drawContours(im, contour3, -1, (0, 0, 255), 2)
        cv2.rectangle(im, (x3, y3), (x3+w3, y3+h3), (50,50,255), 2)

        cv2.imwrite(source_img, im)

        M1 = cv2.moments(contour1)
        print('BLUE: ')
        is_plan = raw_input("1 - plan, 0 - not a plan: ")
        M1['is_plan'] = is_plan
        M1['filename'] = filename
        M1['color'] = 'BLUE'
        print M1
        samples.append(M1)

        M2 = cv2.moments(contour2)
        print('GREEN: ')
        is_plan = raw_input("1 - plan, 0 - not a plan: ")
        M2['is_plan'] = is_plan
        M2['filename'] = filename
        M2['color'] = 'GREEN'
        print M2
        samples.append(M1)

        M3 = cv2.moments(contour3)
        print('RED: ')
        is_plan = raw_input("1 - plan, 0 - not a plan: ")
        M3['is_plan'] = is_plan
        M3['filename'] = filename
        M3['color'] = 'RED'
        print M3
        samples.append(M1)

        continue
    else:
        continue

with open('samples.pickle', 'wb') as handle:
    pickle.dump(samples, handle)

'''sys.exit(0)

if len(sys.argv) != 3:
    print("wrong number of args")
    quit()
'''
