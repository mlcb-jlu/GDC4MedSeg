import cv2
import numpy as np
import scipy.misc
import argparse
import os 
import imageio
from PIL import Image
# from skimage import data,filters,segmentation,measure,morphology
import matplotlib.pyplot as plt



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", default="./results", type=str)

    parser.add_argument("--data_idx", default="brats1", type=str)
    parser.add_argument("--exp_idx", default="brats_43_1", type=str)
    parser.add_argument("--epoch", default="20", type=str)
    parser.add_argument("--phase", default="test", type=str)
    parser.add_argument("--crf_t", default="none", type=str)
    parser.add_argument("--data_species", default="brats", type=str,help="[brats] or [isic] or [covid]")
    parser.add_argument("--kernel_size", default=3, type=int)
    parser.add_argument("--bool_erode", default=1, type=int)
    parser.add_argument("--crf_h", default=10, type=int)#crf range 的上限
    parser.add_argument("--crf_l", default=1, type=int)#crf range 的下限



    args = parser.parse_args()

    # if args.data_species not in ["brats","isic","covid"]:
    #     print("please input the correct data species([brats],[isic] or [covid])!")
    #     exit()

    t_list = []
    #如果不设置这个参数，则为默认的范围网格搜寻，如果指定这个参数，就为指定的crf值
    if args.crf_t=="none":

        # t_list = [1,2,3,4,5,6,7,8,9,10]
        t_list = range(args.crf_l,args.crf_h+1)
            
    else:
        t_list.append(int(args.crf_t))
        # t_list.append(float(args.crf_t))
        print(t_list)

    for crf_t in t_list:
        print("crf_t:{}".format(crf_t))
        #输入路径 crf
        crf_dir = os.path.join(args.result_path,args.exp_idx,args.phase + "_" + args.epoch + "_crf",str(crf_t))
        #输入的最终路径 str(crf_t)+"_res"
        #用于最后进行测试指标的指定dir
        save_dir = os.path.join(args.result_path,args.exp_idx,args.phase + "_" + args.epoch + "_crf",str(crf_t)+"_res")

        #dir1 和dir2 都是中间结果路径，分别是eorde&&dilate和findContours，其中findcontours是isic单独使用的
        save_dir1 = os.path.join(args.result_path,args.exp_idx,args.phase + "_" + args.epoch + "_crf",str(crf_t)+"_erode")
        
        if not os.path.isdir(save_dir1):
            os.mkdir(save_dir1)

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        if args.data_species == "isic":
            save_dir2 = os.path.join(args.result_path,args.exp_idx,args.phase + "_" + args.epoch + "_crf",str(crf_t)+"_findcontours")
            if not os.path.isdir(save_dir2):
                os.mkdir(save_dir2)
        


        crf_lists = os.listdir(crf_dir)
        for idx,img in enumerate(crf_lists) :
            print(str(idx) + " " + img)
            if args.bool_erode == 1:
                #eorde&&dilate
                print("erode and dilating")
                src = cv2.imread(crf_dir + "/" + img, cv2.IMREAD_GRAYSCALE) /255
                kernel = np.ones((args.kernel_size,args.kernel_size),np.uint8)
                erosion = cv2.erode(src,kernel)
                dst = cv2.dilate(erosion,kernel) 
                # scipy.misc.imsave(save_dir+ "/" +img, dst)
                imageio.imsave(save_dir1+ "/" +img, dst*255)
                #isic need findContours
                # if args.data_species == "isic":
                #     ret, thresh = cv2.threshold(dst, 0.5, 1,cv2.THRESH_BINARY)
                #     thresh = np.array(thresh,np.uint8)
                #     # thresh = np.reshape(thresh,(1,256,256))
                #     # cv2.findContours. opencv3版本会返回3个值，opencv2和4只返回后两个值
                #     # img3是返回的二值图像，contours返回的是轮廓像素点列表，一张图像有几个目标区域就有几个列表值
                #     contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                #     sum = []
                #     SUM = SUM1 = -1
                #     idx = -1
                #     for i in range(len(contours)):
                #         area = cv2.contourArea(contours[i])
                #         # print("area is {}/n i is {}",area,i)
                #         sum.append(area)
                #         sum.sort() #列表值从小到大排序
                #         SUM1 = sum[-1] #sum1总是目标面积最大值
                #         # print("max contours is {}",SUM1)
                #         if SUM1 != SUM:
                #             idx = i
                #             SUM = SUM1
                #             # print("max contours idx is {}",i)

                #     for i in range(len(contours)):
                #         if i != idx:
                #             cv2.drawContours(thresh, contours, i, 0, -1)
                #         else:
                #             continue

                #     # scipy.misc.imsave(save_dir2+ "/" +img, thresh*255)
                #     imageio.imsave(save_dir2+ "/" +img, thresh*255)
                #     dst = thresh
                #     #save to res dir
                # # scipy.misc.imsave(save_dir + "/" +img, dst*255)
            else:
                dst=cv2.imread(crf_dir + "/" + img, cv2.IMREAD_GRAYSCALE) /255
            imageio.imsave(save_dir + "/" +img, dst*255)




    

 
