#将mask和1-cam相乘

import os
import argparse
import imageio
from findContours import findCont
import numpy as np

if __name__ == '__main__':
    #去掉mask背景的灰色，变为全黑
    #findcont 分水岭算法，未使用
    parser = argparse.ArgumentParser()
    #parser.add_argument("--cam_img", default="/home/dw/tangsy/attention_gan_seam/AttentionGAN-master/datasets", type=str)
    parser.add_argument("--results", default="/home/dw/tangsy/attention_gan_seam/AttentionGAN-master/results", type=str)

    parser.add_argument("--data_idx", default="brats_1_demo", type=str)
    parser.add_argument("--exp_idx", default="brats_1_31_demo", type=str)
    parser.add_argument("--epoch", default="5", type=str)
    parser.add_argument("--phase", default="val", type=str)

    parser.add_argument("--mask_name", default="a2_b", type=str)
    parser.add_argument("--cam_name", default="G_A_A2B_cam", type=str)
    parser.add_argument("--result_name", default="mask_cam", type=str)

    args = parser.parse_args()

    cam_path = os.path.join(args.results,args.exp_idx,args.phase+"_"+args.epoch+"_"+args.cam_name)#

    mask_path = os.path.join(args.results,args.exp_idx,args.phase+"_"+args.epoch+"_"+args.mask_name)
    result_path = os.path.join(args.results,args.exp_idx,args.phase+"_"+args.epoch+"_"+args.result_name)

    if not os.path.isdir(mask_path):
        os.mkdir(mask_path)
    if not os.path.isdir(result_path):
        os.mkdir(result_path)


    for img in os.listdir(mask_path):

        mask_img = imageio.imread(os.path.join(mask_path,img))/255
        result_img = 1-mask_img

        
        imageio.imsave(os.path.join(result_path,img[:-9]+".png"), result_img*255)






    
