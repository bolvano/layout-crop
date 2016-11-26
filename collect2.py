import numpy as np
import cv2
import sys
import os
import pickle

source_img_directory = './teach_cds_pics'
target_img_directory = './teach_result_pics'
data_directory = './data'

choices = {48: 0, 49: 1}

samples = []

with open(data_directory + '/average_normed_hist.pickle', 'rb') as handle:
    average_normed_hist = pickle.load(handle)

for filename in os.listdir(source_img_directory):
    if filename.endswith(".jpg"):
        print(os.path.join(source_img_directory, filename))

        source_img = os.path.join(source_img_directory, filename)

        vertical_margin = 10
        horizontal_margin = 10

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

        sorteddata = sorteddata[:5]

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

            sample['hist_correl'] = cv2.compareHist(hist/np.linalg.norm(hist),average_normed_hist ,cv2.HISTCMP_CORREL)
            sample['hist_chisqr'] = cv2.compareHist(hist/np.linalg.norm(hist),average_normed_hist ,cv2.HISTCMP_CHISQR)
            sample['hist_bhatt'] = cv2.compareHist(hist/np.linalg.norm(hist),average_normed_hist ,cv2.HISTCMP_BHATTACHARYYA)
            sample['hist_inter'] = cv2.compareHist(hist/np.linalg.norm(hist),average_normed_hist ,cv2.HISTCMP_INTERSECT)

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

            print(c_name + ': ')

            cv2.namedWindow(c_name, cv2.WINDOW_NORMAL)
            cv2.startWindowThread()

            cv2.imshow(c_name, crop_img)

            choice = 0
            while choice not in choices.keys():
                print("1 - plan, 0 - not a plan: ")
                choice = cv2.waitKey(0)

            cv2.destroyAllWindows()

            sample['is_plan'] = choices[choice]
            samples.append(sample)

            with open(data_directory + '/samples_dict_list_ext.pickle', 'wb') as handle:
                pickle.dump(samples, handle)

        continue
    else:
        continue



'''sys.exit(0)

if len(sys.argv) != 3:
    print("wrong number of args")
    quit()
'''