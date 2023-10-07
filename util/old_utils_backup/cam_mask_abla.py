#将mask和1-cam相乘

import os
import argparse
import imageio
from findContours import findCont
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="/home/weidu/tangsy/attentiongan_master/results", type=str)

    parser.add_argument("--data_idx", default="brats_1_demo", type=str)
    parser.add_argument("--exp_idx", default="brats_1_31_demo", type=str)
    parser.add_argument("--epoch", default="5", type=str)
    parser.add_argument("--phase", default="val", type=str)

    parser.add_argument("--a2_b_name", default="a2_b", type=str)
    parser.add_argument("--cam_name", default="G_A_A2B_cam", type=str)
    parser.add_argument("--result_name", default="mask_cam", type=str)

    args = parser.parse_args()

    cam_path = os.path.join(args.results,args.exp_idx,args.phase+"_"+args.epoch+"_"+args.cam_name)#

    a2_b_path = os.path.join(args.results,args.exp_idx,args.phase+"_"+args.epoch+"_"+args.a2_b_name)
    result_path = os.path.join(args.results,args.exp_idx,args.phase+"_"+args.epoch+"_"+args.result_name)



    if not os.path.isdir(cam_path):#
        os.mkdir(cam_path)#

    if not os.path.isdir(a2_b_path):
        os.mkdir(a2_b_path)
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    #without cam just a2_b
    for img in os.listdir(a2_b_path):

        mask_img = imageio.imread(os.path.join(a2_b_path,img))/255
        result_img = mask_img
        result_img = 1-result_img
        imageio.imsave(os.path.join(result_path,img[:-9]+".png"), (result_img*255).astype('uint8'))
    # cam and  a2_b
    # for img in os.listdir(cam_path):#
    #     cam_img = imageio.imread(os.path.join(cam_path,img))/255#
    #     mask_img = imageio.imread(os.path.join(mask_path,img[:-16]+"_"+args.mask_name+".png"))/255#
    #     result_img = cam_img*mask_img#
    #     result_img = 1-result_img
    #     imageio.imsave(os.path.join(result_path,img[:-16]+".png"), (result_img*255).astype('uint8'))#




    
