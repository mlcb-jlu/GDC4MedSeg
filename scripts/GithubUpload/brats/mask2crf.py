
import numpy as np
import scipy.misc
import os.path
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from scipy.signal import convolve2d
import imageio
import argparse


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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_img_path", default="./datasets", type=str)
    parser.add_argument("--result_path", default="./results", type=str)

    parser.add_argument("--data_idx", default="brats1", type=str)
    parser.add_argument("--exp_idx", default="brats_44_1", type=str)
    parser.add_argument("--epoch", default="5", type=str)
    parser.add_argument("--phase", default="val", type=str)
    parser.add_argument("--crf_t", default="none", type=str)#指定某一个crf值
    parser.add_argument("--maskname", default="fused", type=str)

    #进行crf的cam名称是否加maskname的后缀，默认不加，fused的命名规则就是原图像的名称，当需要使用xxxa2_b.png之类的图像进行crf，需要设为true
    parser.add_argument("--suffix", default=False, type=bool)
    parser.add_argument("--crf_h", default=10, type=int)#crf range 的上限
    parser.add_argument("--crf_l", default=1, type=int)#crf range 的下限


    args = parser.parse_args()
    stage = '1'

    for stage in range(1,2):
        orig_img_path = os.path.join(args.orig_img_path,args.data_idx,args.phase+"A")
        # pre_path = os.path.join(args.result_path,args.exp_idx,args.phase+"_"+args.epoch+"_margin/")
        pre_path = os.path.join(args.result_path,args.exp_idx,args.phase+"_"+args.epoch+"_"+args.maskname)
        # pre_path = os.path.join(args.result_path,args.exp_idx,args.phase+"_"+args.epoch+"_mask_cam")
        # pre_path = os.path.join(args.result_path,args.exp_idx,args.phase+"_"+args.epoch+"_G_A_A2B_cam")
        out_cam_pred_path = os.path.join(args.result_path,args.exp_idx,args.phase+"_"+args.epoch+"_crf")
        if not os.path.isdir(out_cam_pred_path):
            os.mkdir(out_cam_pred_path)

        pre_list = os.listdir(orig_img_path)

        for idx,filename in enumerate(pre_list) :

            image_name = filename.split(".png")[0]
            print(str(idx+1) + image_name)
            image_path = os.path.join(orig_img_path, filename)
            orig_img = cv2.imread(image_path)
            orig_img = cv2.resize(orig_img,(256,256))
            orig_img_size = orig_img.shape

            index = filename.rfind(".png")

            if args.suffix is True:
                # our_cam = 1 - cv2.imread(os.path.join(pre_path,image_name+"_"+args.maskname+".png"), cv2.IMREAD_GRAYSCALE) /255
                our_cam = cv2.imread(os.path.join(pre_path,image_name+"_"+args.maskname+".png"), cv2.IMREAD_GRAYSCALE) /255
            else:#不加后缀  fused  病灶区域有值，为白色，周围无值，为黑色  crf前需要置反
                our_cam = 1-cv2.imread(os.path.join(pre_path,image_name+".png"), cv2.IMREAD_GRAYSCALE) /255
            
            
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

            if out_cam_pred_path is not None:
                bg_score = [np.ones_like(norm_cam[0])*0.5]
                pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)*255
                if not os.path.exists(out_cam_pred_path + "/CAM"):
                    os.makedirs(out_cam_pred_path + "/CAM")
                # scipy.misc.imsave(os.path.join(out_cam_pred_path+"/CAM", image_name + '.png'), norm_cam[1])
                imageio.imsave(os.path.join(out_cam_pred_path+"/CAM", image_name + '.png'), norm_cam[1].astype(np.uint8))
                # imageio.imsave(os.path.join(args.out_cam_pred, img_name + '.png'), pred.astype(np.uint8))

            bg_th = 0
            t_list = []
            if args.crf_t=="none":
                t_list = range(args.crf_l,args.crf_h+1)
                # t_list=[1,2,3,4,5,6,7,8,9,10]
            else:
                t_list.append(int(args.crf_t))
                # t_list.append(float(args.crf_t))
                print(t_list)
            
            for t in t_list:#,12,14,16,18,20,22,24,26,28,30,32]:
                    crf = _crf_with_alpha(cam_dict, t)

                    for i in range(256):
                        for j in range(256):
                            if(i+j<=2*bg_th):
                                crf[1][i][j] = 0
                            if (256-i + j <= 2 * bg_th):
                                crf[1][i][j] = 0
                            if (256-j + i <= 2 * bg_th):
                                crf[1][i][j] = 0
                            if (i + j <= 2 * bg_th):
                                crf[1][i][j] = 0
                            if (i + j >= 256*2-bg_th*2):
                                crf[1][i][j] = 0

                    if not os.path.exists(out_cam_pred_path+"/"+str(t)):
                        os.makedirs(out_cam_pred_path+"/"+str(t))
                    # scipy.misc.imsave(os.path.join(out_cam_pred_path+"/"+str(t), image_name + '.png'), crf[1] * 255)
                    imageio.imsave(os.path.join(out_cam_pred_path+"/"+str(t), image_name + '.png'), (crf[1] * 255).astype(np.uint8))


