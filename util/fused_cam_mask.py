import os
import cv2
import imageio
import numpy as np


if __name__ == '__main__':

    result_dir = "/home/weidu/tangsy/attentiongan_master/results"
    exp_name = "n_brats_rn1_d5_i4"
    iter = 125
    pahse = "test"
    cam_path = os.path.join(result_dir,exp_name,pahse+'_'+str(iter)+'_G_A_A2B_cam')
    a2b_path = os.path.join(result_dir,exp_name,pahse+'_'+str(iter)+'_a2_b')
    result_path = os.path.join(result_dir,exp_name,pahse+'_'+str(iter)+'_fused')


        
    #create dir
    if not os.path.isdir(cam_path):
        os.mkdir(cam_path)
        print("{} is created successfully".format(cam_path))
    if not os.path.isdir(a2b_path):
        os.mkdir(a2b_path)
        print("{} is created successfully".format(a2b_path))
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
        print("{} is created successfully".format(result_path))

    #return fused list
    fused_list = []

    for idx,img in enumerate(os.listdir(a2b_path)):

        cam_img = imageio.imread(os.path.join(cam_path,img[:-9]+"_G_A_A2B_cam.png"))/255
        a2b_img = imageio.imread(os.path.join(a2b_path,img))/255
        result_img = cam_img + a2b_img
        result_img = 1-result_img
        fused_list.append(result_img)
        
        imageio.imsave(os.path.join(result_path,img[:-9]+".png"), (result_img * 255).astype(np.uint8))

    print("fused img number is {},all of them have been saved".format(idx+1))
