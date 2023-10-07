import os
import cv2
from shutil import copyfile

#处理原图

# isic_path = "/home/dw/tangsy/attention_gan_seam/AttentionGAN-master/datasets/isic_new/trainA"
isic_path = "/home/dw/tangsy/attention_gan_seam/AttentionGAN-master/datasets/isic_new/valA_label"
out_path = "/home/dw/tangsy/attention_gan_seam/AttentionGAN-master/datasets/isic_new/valA_label_resize"
if not os.path.isdir(out_path):
        os.mkdir(out_path)
img_list = os.listdir(isic_path)
for idx,i in enumerate(img_list) :
    print(str(idx)+" "+i)
    img = cv2.imread(os.path.join(isic_path,i))
    out_img = os.path.join(out_path,i)[:-4]+".png"
    # res_img = cv2.resize(img,(256,256),interpolation=cv2.INTER_CUBIC)
    res_img = cv2.resize(img,(256,256),interpolation=cv2.INTER_NEAREST)#最近邻插值
    # res_img = cv2.cvtColor(res_img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(out_img, res_img)

#切片

# isic_path = "/home/dw/tangsy/attention_gan_seam/AttentionGAN-master/datasets/isic_new/trainA"
# out_path = "/home/dw/tangsy/attention_gan_seam/AttentionGAN-master/datasets/isic_new/trainA_split"
# if not os.path.isdir(out_path):
#         os.mkdir(out_path)
# row = 5
# column = 5

# img_list = os.listdir(isic_path)
# for idx,im in enumerate(img_list) :
#     print(str(idx)+" "+im)
#     img = cv2.imread(os.path.join(isic_path,im))
#     height, width = img.shape[:2]
#     # print('height %d widht %d' % (height, width))

#     row_step = (int)(height/row)
#     column_step = (int)(width/column)

#     # print('row step %d col step %d'% (row_step, column_step))

#     # print('height %d widht %d' % (row_step*row, column_step*column))

#     img = img[0:row_step*row, 0:column_step*column]

#     for i in range(row):
#         for j in range(column):
#             pic_name = os.path.join(out_path,im[:-4] + "_" +str(i) + "_" + str(j) + ".png")
#             tmp_img = img[(i*row_step):(i*row_step+row_step), (j*column_step):(j*column_step)+column_step]
#             tmp_img = cv2.resize(tmp_img,(256,256),interpolation=cv2.INTER_CUBIC)

#             cv2.imwrite(pic_name, tmp_img)


#挑选
# isic_path = "/home/dw/tangsy/attention_gan_seam/AttentionGAN-master/datasets/isic_new/trainA_split"
# out_path = "/home/dw/tangsy/attention_gan_seam/AttentionGAN-master/datasets/isic_new/trainA_split_pickup"

# if not os.path.isdir(out_path):
#     os.mkdir(out_path)
# img_list = os.listdir(isic_path)
# for idx,im in enumerate(img_list):
#     if "_0_2" in im :
#         print(str(idx)+" "+im)
#         src_img0 = os.path.join(isic_path,im[:-8]+"_0_1.png")
#         tar_img0 = os.path.join(out_path,im[:-8]+"_normal_0.png")
#         copyfile(src_img0,tar_img0)

#     if "_4_2" in im :
#         print(str(idx)+" "+im)
#         src_img1 = os.path.join(isic_path,im[:-8]+"_4_1.png")
#         tar_img1 = os.path.join(out_path,im[:-8]+"_normal_1.png")
#         copyfile(src_img1,tar_img1)



#随机挑选

# num = 2000
# import random
# path="/home/dw/tangsy/attention_gan_seam/AttentionGAN-master/datasets/isic_new/trainA_split_pickup"
# tar = "/home/dw/tangsy/attention_gan_seam/AttentionGAN-master/datasets/isic_new/trainB"
# imgs = []
# for x in os.listdir(path):
#     if x.endswith('png'):
# 	    imgs.append(x)
# selected_imgs=random.sample(imgs,k=num)
# print(selected_imgs)

# from shutil import copyfile
# for img in selected_imgs:
#     src=os.path.join(path,img)
#     dst=os.path.join(tar,img)    
#     copyfile(src,dst)
# print("copy done")






