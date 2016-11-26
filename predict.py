import numpy as np
import cv2
import sys
import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

source_img_directory = './predict_cds_pics'
target_img_directory = './predict_result_pics'
data_directory = './data'

samples = []

for filename in os.listdir(source_img_directory):
    if filename.endswith(".jpg"):
        print(os.path.join(source_img_directory, filename))

        source_img = os.path.join(source_img_directory, filename)

        im = cv2.imread(source_img)

        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        im_neg = (255-imgray)
        ret,thresh = cv2.threshold(im_neg,1,255,0)

        cv2.imwrite(os.path.join(target_img_directory, filename)[:-4] + '_thresh' + '.jpg', thresh)

        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        areaArray = []

        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            areaArray.append(area)

        sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

        sorteddata = sorteddata[:10]

        i = 0
        for c in sorteddata:
            c = c[1]
            i += 1

            c_name = filename[:-4] + '_' + str(i) + '.jpg'

            sample = cv2.moments(c)

            if sample['m00'] == 0:
                print ("Empty contour, skipping...\n")
                continue

            sample['contour'] = c_name

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

            cv2.imwrite(os.path.join(target_img_directory, c_name), crop_img)

            print(c_name)

            samples.append(sample)

        continue
    else:
        continue

df = pd.DataFrame(samples)
df = df.set_index(['contour'])

features = ['avc1', 'avc2', 'avc3', 'cx', 'cy', 'extent', 'm00', 'rect_extent', 'top_bin1', 'top_bin2', 'top_bin3', 'top_bin4', 'top_bin5', 'white_balance']

x = df[features]

with open(data_directory+'/rf_wo_moments2.pickle', 'rb') as handle:
    rf = pickle.load(handle)

df['predict'] = rf.predict(x)
proba = rf.predict_proba(x)
df['prob0'], df['prob1'] = np.hsplit(rf.predict_proba(x),2)[0], np.hsplit(rf.predict_proba(x),2)[1]

print(df[['predict', 'prob0', 'prob1']][df['predict']==1])
