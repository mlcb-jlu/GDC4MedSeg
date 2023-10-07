import os
import argparse
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sourse_path", default="/home/weidu/tangsy/attentiongan_master/results", type=str)
    parser.add_argument("--exp_idx", default="brats_1_3_demo", type=str)
    parser.add_argument("--epoch", default="60", type=str)
    parser.add_argument("--phase", default="val", type=str)
    parser.add_argument("--tar_name", default="a1_b", type=str)

    args = parser.parse_args()
    s_epoch_path = args.phase+"_"+args.epoch
    t_epoch_path = args.phase+"_"+args.epoch+"_"+args.tar_name
    img_path = os.path.join(args.sourse_path,args.exp_idx,s_epoch_path)
    target_path = os.path.join(args.sourse_path,args.exp_idx,t_epoch_path)
    if not os.path.isdir(target_path):
        os.mkdir(target_path)
    for img in os.listdir(os.path.join(img_path,"images")):
        if args.tar_name in img:
            shutil.copy(os.path.join(img_path,"images",img),os.path.join(target_path,img))
            print(img)
    print("Copy of {} has been completed!".format(args.tar_name))