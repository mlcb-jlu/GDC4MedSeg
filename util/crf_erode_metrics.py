
import numpy as np
import scipy.misc
import os.path
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from scipy.signal import convolve2d
import imageio
import argparse
from five_metrics_crf_deal import five_m

def crf_inference(img, probs, t=10, scale_factor=1, labels=21):


    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

def _crf_with_alpha(cam_dict, alpha):
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

    n_crf_al = dict()

    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key+1] = crf_score[i+1]

    return n_crf_al

def iou_score(output, target):
    smooth = 1e-7
    # plt_hist(output)
    # plt_hist(target)
    output = output > 0.5  # 大于0.5为TRUE,小于0.5为FALSE
    target = target > 0.5

    intersection = (output & target).sum()
    union = (output | target).sum()

    TP = float(np.sum(np.logical_and(output == True, target == True)))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = float(np.sum(np.logical_and(output == False, target == False)))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = float(np.sum(np.logical_and(output == True, target == False)))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = float(np.sum(np.logical_and(output == False, target == True)))

    #  calculate JA, Dice, SE, SP
    JA = TP / ((TP + FN + FP + 1e-7))
    AC = (TP + TN) / (TP + FP + TN + FN + 1e-7)
    DI = 2 * TP / (2 * TP + FN + FP + 1e-7)
    SE = TP / (TP + FN + 1e-7)
    SP = TN / ((TN + FP + 1e-7))

    return (intersection + smooth) / (union + smooth),JA,AC,DI,SE,SP


def five_metrics(pre_path,label_path):
    count = 0
    iou_sum = 0
    JA_sum = 0
    AC_sum = 0
    DI_sum = 0
    SE_sum = 0
    SP_sum = 0
    pre_list = os.listdir(pre_path)
    for filename in pre_list:
        count += 1
        image_path = os.path.join(pre_path, filename)
        # pre = cv2.imread(image_path)
        pre = imageio.imread(image_path)

        index = filename.rfind(".")
        # label_name = filename[:index] + ".png"
        label_name = filename[:index] + "_segmentation.png"
        target_path = os.path.join(label_path, label_name)
        # target = cv2.imread(target_path)
        target = imageio.imread(target_path)

        iou,JA,AC,DI,SE,SP = iou_score(pre/255, target[:,:,0]/255)
        iou_sum += iou
        JA_sum += JA
        AC_sum += AC
        DI_sum += DI
        SE_sum += SE
        SP_sum += SP

        print(filename,JA)

    print("JA:%.4f " % (JA_sum/count))
    print("AC:%.4f " % (AC_sum/count))
    print("DI:%.4f " % (DI_sum/count))
    print("SE:%.4f " % (SE_sum/count))
    print("SP:%.4f " % (SP_sum/count))
    list_result = [JA_sum/count, AC_sum/count, DI_sum/count, SE_sum/count ,SP_sum/count]
    list_result_round = np.round(list_result, 4)
    print("  JA     AC     DI      SE     SP")
    print(", ".join(str(i) for i in list_result_round))


def five_metrics1(pre_path,label_path):
    count = 0
    iou_sum = 0
    JA_sum = 0
    AC_sum = 0
    DI_sum = 0
    SE_sum = 0
    SP_sum = 0
    pre_list = os.listdir(pre_path)
    for filename in pre_list:
        count += 1
        image_path = os.path.join(pre_path, filename)
        # pre = cv2.imread(image_path)
        pre = imageio.imread(image_path)

        index = filename.rfind("r")
        # label_name = filename[:index] + ".png"
        label_name = filename[:index-1] + "_segmentation.png"
        target_path = os.path.join(label_path, label_name)
        # target = cv2.imread(target_path)
        target = imageio.imread(target_path)

        iou,JA,AC,DI,SE,SP = iou_score(pre/255, target/255)
        iou_sum += iou
        JA_sum += JA
        AC_sum += AC
        DI_sum += DI
        SE_sum += SE
        SP_sum += SP

        print(filename,JA)

    print("JA:%.4f " % (JA_sum/count))
    print("AC:%.4f " % (AC_sum/count))
    print("DI:%.4f " % (DI_sum/count))
    print("SE:%.4f " % (SE_sum/count))
    print("SP:%.4f " % (SP_sum/count))
    list_result = [JA_sum/count, AC_sum/count, DI_sum/count, SE_sum/count ,SP_sum/count]
    list_result_round = np.round(list_result, 4)
    print("  JA     AC     DI      SE     SP")
    print(", ".join(str(i) for i in list_result_round))






if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_img_path", default="/home/weidu/tangsy/attentiongan_master/datasets/covid_semi_50_new_attentiongan", type=str)
    parser.add_argument("--result_path", default="/home/weidu/tangsy/attentiongan_master/results", type=str)

    #共有参数
    parser.add_argument("--data_idx", default="covid_1", type=str)
    parser.add_argument("--exp_idx", default="n_covid_semi_50_rn1_d1_i6", type=str)
    parser.add_argument("--epoch", default="50", type=str)
    parser.add_argument("--phase", default="test", type=str)
    parser.add_argument("--maskname", default="fused", type=str)
    parser.add_argument("--mask_back_black", default=True, type=bool)##背景为黑色，不需要置反

    
    parser.add_argument("--crf_t", default="none", type=str)#指定某一个crf值
    parser.add_argument("--crf_h", default=12, type=int)#crf range 的上限
    parser.add_argument("--crf_l", default=1, type=int)#crf range 的下限

    #crf
    parser.add_argument("--suffix", default=False, type=bool)#进行crf的cam名称是否加maskname的后缀，默认不加，fused的命名规则就是原图像的名称，当需要使用xxxa2_b.png之类的图像进行crf，需要设为true
    

    #erode
    parser.add_argument("--kernel_size", default=3, type=int)
    parser.add_argument("--bool_erode", default=1, type=int)


    args = parser.parse_args()

    orig_img_path = os.path.join(args.orig_img_path,args.data_idx,args.phase+"A")
    labels_path = os.path.join(args.orig_img_path,args.data_idx,args.phase+"A_label")
    # pre_path = os.path.join(args.result_path,args.exp_idx,args.phase+"_"+args.epoch+"_margin/")
    pre_path = os.path.join(args.result_path,args.exp_idx,args.phase+"_"+args.epoch+"_"+args.maskname)
    # pre_path = os.path.join(args.result_path,args.exp_idx,args.phase+"_"+args.epoch+"_mask_cam")
    # pre_path = os.path.join(args.result_path,args.exp_idx,args.phase+"_"+args.epoch+"_G_A_A2B_cam")
    out_cam_pred_path = os.path.join(args.result_path,args.exp_idx,args.phase+"_"+args.epoch+"_crf")

    erode_save_path = os.path.join(args.result_path,args.exp_idx,args.phase+"_"+args.epoch+"_erode")
    if not os.path.isdir(out_cam_pred_path):
        os.mkdir(out_cam_pred_path)

    if not os.path.isdir(erode_save_path):
        os.mkdir(erode_save_path)

    pre_list = os.listdir(orig_img_path)

    for idx,filename in enumerate(pre_list) :

        image_name = filename.split(".png")[0]
        print(str(idx+1) + image_name)
        image_path = os.path.join(orig_img_path, filename)
        orig_img = cv2.imread(image_path)
        orig_img = cv2.resize(orig_img,(256,256))
        orig_img_size = orig_img.shape

        label_path = os.path.join(labels_path, image_name+"_segmentation.png")
        label_img = cv2.imread(label_path)

        index = filename.rfind(".png")

        if args.suffix is True:
            our_cam = cv2.imread(os.path.join(pre_path,image_name+"_"+args.maskname+".png"), cv2.IMREAD_GRAYSCALE) /255
        else:
            our_cam = cv2.imread(os.path.join(pre_path,image_name+".png"), cv2.IMREAD_GRAYSCALE) /255
        
        
        cam_list=list()
        our_cam = convolve2d(our_cam, np.ones((7, 7)), 'same') /49
        #our_cam = np.where(our_cam < 0.95, np.zeros_like(our_cam), our_cam)
        #our_cam = np.where(our_cam >= 0.95, np.ones_like(our_cam), our_cam)

        our_cam = np.reshape(our_cam,[1,256,256])
        our_cam = np.concatenate((our_cam,1 - our_cam),axis=0)
        cam_list.append(our_cam)

        sum_cam = np.sum(cam_list, axis=0)
        norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)

        cam_dict = {}
        for i in range(0,1):
            cam_dict[i] = norm_cam[i+1]

        # bg_th = 0
        # t_list = []
        # if args.crf_t=="none":
        #     t_list = range(args.crf_l,args.crf_h+1)
        #     # t_list=[1,2,3,4,5,6,7,8,9,10]
        # else:
        #     t_list.append(int(args.crf_t))
        #     # t_list.append(float(args.crf_t))
        #     print(t_list)

        # max_JA=0
        # crf_t=0
        # crf_metrix_list = []
        # result_crf=None
        
        # for t in t_list:#,12,14,16,18,20,22,24,26,28,30,32]:
        #         crf = _crf_with_alpha(cam_dict, t)

        #         for i in range(256):
        #             for j in range(256):
        #                 if(i+j<=2*bg_th):
        #                     crf[1][i][j] = 0
        #                 if (256-i + j <= 2 * bg_th):
        #                     crf[1][i][j] = 0
        #                 if (256-j + i <= 2 * bg_th):
        #                     crf[1][i][j] = 0
        #                 if (i + j <= 2 * bg_th):
        #                     crf[1][i][j] = 0
        #                 if (i + j >= 256*2-bg_th*2):
        #                     crf[1][i][j] = 0

                
        #         src = crf
        #         kernel = np.ones((args.kernel_size,args.kernel_size),np.uint8)
        #         erosion = cv2.erode(src[1],kernel)
        #         erode_dst = cv2.dilate(erosion,kernel) 
        #         # print(label_img)
        #         # print(erode_dst.shape)
        #         iou,JA,AC,DI,SE,SP = iou_score(erode_dst, (label_img[:,:,0])/255)
        #         if JA>max_JA:
        #             max_JA=JA
        #             crf_t=t
        #             result_crf=crf
        #             result_erode_dst = erode_dst

        # imageio.imsave(os.path.join(out_cam_pred_path, image_name + '.png'), (crf[1] * 255).astype(np.uint8))
        # imageio.imsave(os.path.join(erode_save_path, image_name + '.png'),(result_erode_dst*255).astype(np.uint8))
        # crf_metrix_list.append([image_name,max_JA,crf])
        # with open(os.path.join(out_cam_pred_path,"results.txt"), "a+") as f:
        #     f.write(filename + ':      max_JA:'+str(max_JA)+ '           crf:'+str(crf_t)+'\n')

        # five_metrics(erode_save_path,labels_path)
        five_metrics1(pre_path,labels_path)

        

            

            
            
            # scipy.misc.imsave(save_dir+ "/" +img, dst)
            






