
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse


def plt_hist(img):
    plt.hist(img.ravel(), 256, [0, 1])
    plt.show()
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

def five_m(folder_path, label_path):
    folder_list = os.listdir(folder_path)

    for folder in folder_list:
        count = 0
        iou_sum = 0
        JA_sum = 0
        AC_sum = 0
        DI_sum = 0
        SE_sum = 0
        SP_sum = 0
        pre_list = os.listdir(os.path.join(folder_path, folder))
        for filename in pre_list:
            count += 1
            image_path = os.path.join(folder_path, folder, filename)
            pre = cv2.imread(image_path)

            index = filename.rfind(".")
            label_name = filename[:index] + "_segmentation.png"
            target_path = os.path.join(label_path, label_name)
            target = cv2.imread(target_path)

            iou,JA,AC,DI,SE,SP = iou_score(pre/255, target/255)
            iou_sum += iou
            JA_sum += JA
            AC_sum += AC
            DI_sum += DI
            SE_sum += SE
            SP_sum += SP

        # print(iou_sum/count,JA_sum/count,AC_sum/count,DI_sum/count,SE_sum/count,SP_sum/count)
        print("folder:%s" % (folder))
        print("JA:%.4f " % (JA_sum/count))
        print("AC:%.4f " % (AC_sum/count))
        print("DI:%.4f " % (DI_sum/count))
        print("SE:%.4f " % (SE_sum/count))
        print("SP:%.4f " % (SP_sum/count))
        list_result = [JA_sum/count, AC_sum/count, DI_sum/count, SE_sum/count ,SP_sum/count]
        list_result_round = np.round(list_result, 4)
        print("  JA     AC     DI      SE     SP")
        print(", ".join(str(i) for i in list_result_round))
        if (float(list_result_round[0]) > 0.63 and float(list_result_round[0]) < 1):
            with open(folder_path + "results.txt", "a+") as f:
                f.write(folder + '\n')
                # f.write('\n')
                f.write(str(list_result))
                f.write('\n')
                f.write('==' * 5)
                f.write('\n')
        return list_result_round

def five_single_folder(args,img_path, label_path):
        count = 0
        iou_sum = 0
        JA_sum = 0
        AC_sum = 0
        DI_sum = 0
        SE_sum = 0
        SP_sum = 0
        for filename in os.listdir(img_path):
            if "coronacases_001" not in filename:
                count += 1
                image_path = os.path.join(img_path, filename)
                pre = cv2.imread(image_path)

                index = filename.rfind(".")
                # label_name = filename[:index] + "_segmentation.png"
                label_name = filename[:index] + ".png"
                target_path = os.path.join(label_path, label_name)
                target = cv2.imread(target_path)

                if pre is None or  target is None:
                    count -= 1
                    continue
                iou,JA,AC,DI,SE,SP = iou_score(pre/255, target/255)
                print("{} iou:{}".format(filename,iou))
                print("JA:{} AC:{} DI:{} SE:{} SP:{}".format(JA,AC,DI,SE,SP))

                iou_sum += iou
                JA_sum += JA
                AC_sum += AC
                DI_sum += DI
                SE_sum += SE
                SP_sum += SP

        # print(iou_sum/count,JA_sum/count,AC_sum/count,DI_sum/count,SE_sum/count,SP_sum/count)
        print("folder:%s" % (img_path))
        print("JA:%.4f " % (JA_sum/count))
        print("AC:%.4f " % (AC_sum/count))
        print("DI:%.4f " % (DI_sum/count))
        print("SE:%.4f " % (SE_sum/count))
        print("SP:%.4f " % (SP_sum/count))
        list_result = [JA_sum/count, AC_sum/count, DI_sum/count, SE_sum/count ,SP_sum/count]
        list_result_round = np.round(list_result, 4)
        print("  JA     AC     DI      SE     SP")
        print(", ".join(str(i) for i in list_result_round))

        with open(os.path.join(args.result_path,args.exp_idx,args.phase+"_"+args.epoch+"_crf/results.txt"), "a+") as f:
            f.write("folder:%s\n" % (img_path))
            f.write("JA:%.4f \n" % (JA_sum/count))
            f.write("AC:%.4f \n" % (AC_sum/count))
            f.write("DI:%.4f \n" % (DI_sum/count))
            f.write("SE:%.4f \n" % (SE_sum/count))
            f.write("SP:%.4f \n" % (SP_sum/count))
            f.write("  JA     AC     DI      SE     SP")
            f.write(", ".join(str(i) for i in list_result_round))
            # f.write('\n')
            f.write(str(list_result))
            f.write('\n')
            f.write('==' * 5)
            f.write('\n')
        return list_result_round
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_img_path", default="/home/weidu/tangsy/attentiongan_master/datasets", type=str)
    parser.add_argument("--result_path", default="/home/weidu/tangsy/attentiongan_master/results", type=str)


    parser.add_argument("--data_idx", default="brats_1_demo", type=str)
    parser.add_argument("--exp_idx", default="brats_1_3_demo", type=str)
    parser.add_argument("--epoch", default="40", type=str)
    parser.add_argument("--phase", default="val", type=str)
    parser.add_argument("--crf_t", default="none", type=str)
    args = parser.parse_args()

    t_list = []
    if args.crf_t=="none":
        t_list = [1,2,3,4,5,6,7,8,9,10]
        # t_list = [8,9,10,11,12]
    else:
        t_list.append(int(args.crf_t))
        print(t_list)
    gt_path = os.path.join(args.orig_img_path,args.data_idx,args.phase+"A_label")
    crf_path = os.path.join(args.result_path,args.exp_idx,args.phase+"_"+args.epoch+"_crf")
    list_result = []
    for t in t_list:
        t_crf_path = crf_path+"/"+str(t)+"_erode"
        list_result_t = five_single_folder(args,t_crf_path,gt_path)
        list_result.append(list_result_t)

    arr_result = np.array(list_result)
    max_score = np.max(arr_result[:,0])
    max_t = t_list[np.argmax(arr_result[:,0])]
    with open(os.path.join(args.result_path,args.exp_idx,args.phase+"_results.txt"), "a+") as f:
        f.write(args.epoch+ 'erode\n')
        f.write("max_t:"+str(max_t)+'\n')
        f.write("max_scror:"+str(max_score)+'\n')
        f.write("\n-----------------------------------------\n\n")
        



