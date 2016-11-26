import numpy as np
import cv2
import sys
import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

source_img_directory = './all_cds_pics'
target_img_directory = './all_result_pics'
data_directory = './data'

features = ['avc1', 'avc2', 'avc3', 'cx', 'cy', 'extent', 'm00', 'rect_extent', 'top_bin1', 'top_bin2', 'top_bin3', 'top_bin4', 'top_bin5', 'white_balance']

with open(data_directory + '/rf_wo_moments2.pickle', 'rb') as handle:
    rf = pickle.load(handle)

samples = []

for filename in os.listdir(source_img_directory):
    if filename.endswith(".jpg"):
        print(os.path.join(source_img_directory, filename))

        source_img = os.path.join(source_img_directory, filename)

        try:
            im = cv2.imread(source_img)
        except:
            print("INVALID IMAGE, SKIPPING...")
            continue

        try:
            imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        except:
            print("INVALID IMAGE, SKIPPING...")
            continue

        cv2.imwrite(os.path.join(target_img_directory, filename), im)

        im_neg = (255-imgray)
        ret,thresh = cv2.threshold(im_neg,1,255,0)

        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        areaArray = []

        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            areaArray.append(area)

        sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

        sorteddata = sorteddata[:5]

        i = 0
        for c in sorteddata:
            c = c[1]
            i += 1

            sample = cv2.moments(c)

            if sample['m00'] == 0:
                print ("Empty contour, skipping...\n")
                continue

            sample['cx'] = sample['m10']/sample['m00']
            sample['cy'] = sample['m01']/sample['m00']

            x, y, w, h = cv2.boundingRect(c)
            crop_img = im[y: y + h, x: x + w]

            average_color_per_row = np.average(crop_img, axis=0)
            average_color = np.average(average_color_per_row, axis=0)

            sample['avc1'] = float(average_color[0])
            sample['avc2'] = float(average_color[1])
            sample['avc3'] = float(average_color[2])

            # area / rect
            sample['extent'] = sample['m00'] / (h*w)

            #min_rect / rect
            (min_x,min_y),(min_w,min_h),min_theta = cv2.minAreaRect(c)
            sample['rect_extent'] = (min_w*min_h) / (h*w)

            #print(sample['m00'], min_w*min_h, h*w)
            #sys.exit(0)

            hist = cv2.calcHist([crop_img],[0],None,[256],[0,256])
            # 'white_balance'
            #print(int(hist[-1])/sample['m00'])
            sample['white_balance'] = int(hist[-1])/(h*w)

            #N top bins (intensity peaks)
            hist_flat = hist.reshape(256,)
            hist_flat_sorted = hist_flat.argsort()
            N = 5
            for j in range(1, N+1):
                sample['top_bin' + str(j)] = hist_flat_sorted[ j* -1 ]

            print(sample)

            s = pd.Series(sample)
            x = s[features]

            proba = rf.predict_proba(x.reshape(1,-1))
            predict = rf.predict(x.reshape(1,-1))

            if int(predict[0]) == 1:
                c_name = filename[:-4] + '_' + str(i) + "_" + ("%.2f" % proba[0][1]) +'.jpg'
                cv2.imwrite(os.path.join(target_img_directory, c_name), crop_img)

        continue
    else:
        continue

