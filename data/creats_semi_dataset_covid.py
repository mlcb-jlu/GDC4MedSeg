import os
import argparse
import imageio
import numpy as np
import shutil

def random_pick(args,src_data_path):
    src_list = "/home/weidu/tangsy/attentiongan_master/datasets/covid_1/563_A_weak"
    A_weak_563_list = sorted(os.listdir(src_list))
    B_weak_563_list = 
    for img in os.listdir(src_list):

    if not os.path.isdir(os.path.join(src_data_path,"50_513")):
        os.mkdir(os.path.join(src_data_path,"50_513"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/home/dw/tangsy/attention_gan_seam/AttentionGAN-master/datasets", type=str)

    parser.add_argument("--category",default="covid",type=str)
    parser.add_argument("--idx",default="5",type=str)

    
    args = parser.parse_args()

    src_data_path = os.path.join(args.dataset,args.category+"_"+args.idx)
    tar_data_path1 = os.path.join(args.dataset,args.category+"_"+args.idx+"_semi1")
    tar_data_path2 = os.path.join(args.dataset,args.category+"_"+args.idx+"_semi2")



    if not os.path.isdir(tar_data_path1):
        os.mkdir(tar_data_path1)

    if not os.path.isdir(tar_data_path2):
        os.mkdir(tar_data_path2)
    
    train_A = os.path.join(tar_data_path1,'trainA')
    train_B = os.path.join(tar_data_path1,'trainB')
    test_A = os.path.join(tar_data_path1,'testA')
    test_B = os.path.join(tar_data_path1,'testB')
    test_A_label = os.path.join(tar_data_path1,'testA_label')
    val_A = os.path.join(tar_data_path1,'valA')
    val_B = os.path.join(tar_data_path1,'valB')
    val_A_label = os.path.join(tar_data_path1,'valA_label')

    train_A_sup = os.path.join(tar_data_path2,'trainA')
    train_B_sup = os.path.join(tar_data_path2,'trainB')
    train_A_sup_labels = os.path.join(tar_data_path2,'trainA_sup_label')

    # tar_paths = [train_A,train_B,test_A,test_B,test_A_label,val_A,val_B,val_A_label]
    tar_paths = [train_A,train_B,\
                train_A_sup,train_B_sup,train_A_sup_labels,\
                test_A,test_B,test_A_label,\
                val_A,val_B,val_A_label]

    

    src_train_A = os.path.join(src_data_path,'18_182',"fake","images")
    src_train_B = os.path.join(src_data_path,'182_B_semi')
    src_train_B_sup = os.path.join(src_data_path,'182_B_semi')
    src_train_A_sup = os.path.join(src_data_path,'18_182',"real","images")
    src_train_A_sup_label = os.path.join(src_data_path,'18_182',"real","labels")
    
    

    src_test_A = os.path.join(src_data_path,'testA')
    src_test_B = os.path.join(src_data_path,'testB')
    src_test_A_label = os.path.join(src_data_path,'testA_label')
    src_val_A = os.path.join(src_data_path,'valA')
    src_val_B = os.path.join(src_data_path,'valB')
    src_val_A_label = os.path.join(src_data_path,'valA_label')
    src_paths = [src_train_A,src_train_B,\
                src_train_A_sup,src_train_B_sup,src_train_A_sup_label,\
                src_test_A,src_test_B,src_test_A_label,\
                src_val_A,src_val_B,src_val_A_label]


    for idx in range(len(tar_paths)):
        print(src_paths[idx])
        print(tar_paths[idx])
        
        if not os.path.isdir(tar_paths[idx]):
            os.mkdir(tar_paths[idx])
        # print("{} is created".format(tar_paths[idx]))
        if idx==3:#trainB_sup
            for i,img in enumerate(os.listdir(src_paths[idx])) :
                if i<18:
                    shutil.copy(os.path.join(src_paths[idx],img),tar_paths[idx])
                    if (i+1)%10==0:
                        print("{0} imgs  copied".format(i+1))
        else:
            for i,img in  enumerate(os.listdir(src_paths[idx])) :
                    shutil.copy(os.path.join(src_paths[idx],img),tar_paths[idx])
                    if (i+1)%10==0:
                        print("{0} imgs  copied".format(i+1))