import cv2
import numpy as np
import scipy.misc
import argparse
import os 
from PIL import Image
from skimage import data,filters,segmentation,measure,morphology
import matplotlib.pyplot as plt



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    9 
    parser.add_argument("--result_path", default="/home/dw/tangsy/attention_gan_seam/AttentionGAN-master/results", type=str)

    parser.add_argument("--data_idx", default="isic1", type=str)
    parser.add_argument("--exp_idx", default="isic_43_1_6", type=str)
    parser.add_argument("--epoch", default="45", type=str)
    parser.add_argument("--phase", default="test", type=str)
    parser.add_argument("--crf_t", default="none", type=str)

    args = parser.parse_args()

    t_list = []
    if args.crf_t=="none":
        t_list = [1,2,3,4,5,6,7,8,9,10]
    else:
        t_list.append(int(args.crf_t))
        print(t_list)

    for crf_t in t_list:
        print("crf_t:{}".format(crf_t))
        crf_dir = os.path.join(args.result_path,args.exp_idx,args.phase + "_" + args.epoch + "_crf",str(crf_t))
        save_dir1 = os.path.join(args.result_path,args.exp_idx,args.phase + "_" + args.epoch + "_crf",str(crf_t)+"_erode")
        save_dir2 = os.path.join(args.result_path,args.exp_idx,args.phase + "_" + args.epoch + "_crf",str(crf_t)+"_findcontours")
        if not os.path.isdir(save_dir1):
            os.mkdir(save_dir1)
        if not os.path.isdir(save_dir2):
            os.mkdir(save_dir2)

        crf_lists = os.listdir(crf_dir)
        for idx,img in enumerate(crf_lists) :
            print(str(idx) + " " + img)

            #eorde&&dilate
            src = cv2.imread(crf_dir + "/" + img, cv2.IMREAD_GRAYSCALE) /255
            kernel = np.ones((3,3),np.uint8)
            erosion = cv2.erode(src,kernel)
            dst = cv2.dilate(erosion,kernel) 

            scipy.misc.imsave(save_dir1+ "/" +img, dst*255)

            ret, thresh = cv2.threshold(dst, 0.5, 1,cv2.THRESH_BINARY)
            thresh = np.array(thresh,np.uint8)
            # thresh = np.reshape(thresh,(1,256,256))
            # cv2.findContours. opencv3版本会返回3个值，opencv2和4只返回后两个值
            # img3是返回的二值图像，contours返回的是轮廓像素点列表，一张图像有几个目标区域就有几个列表值
            contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            sum = []
            SUM = SUM1 = -1
            idx = -1
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                # print("area is {}/n i is {}",area,i)
                sum.append(area)
                sum.sort() #列表值从小到大排序
                SUM1 = sum[-1] #sum1总是目标面积最大值
                # print("max contours is {}",SUM1)
                if SUM1 != SUM:
                    idx = i
                    SUM = SUM1
                    # print("max contours idx is {}",i)

            for i in range(len(contours)):
                if i != idx:
                    cv2.drawContours(thresh, contours, i, 0, -1)
                else:
                    continue

            scipy.misc.imsave(save_dir2+ "/" +img, thresh*255)

    

 
