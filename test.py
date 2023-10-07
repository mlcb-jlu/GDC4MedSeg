"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html1
import shutil
import imageio
import numpy as np
def mv_img(opt,target_name):
    """
    input:
        opt:命令行参数
        target_name:需要移动的mask的名称，例如“a2_b”，"G_A_A2B_cam""mask_cam"，
    """
    s_epoch_path = opt.phase+"_"+opt.epoch
    t_epoch_path = opt.phase+"_"+opt.epoch+"_"+target_name
    img_path = os.path.join(opt.results_dir,opt.name,s_epoch_path)
    target_path = os.path.join(opt.results_dir,opt.name,t_epoch_path)
    img_num = 0
    if not os.path.isdir(target_path):
        os.mkdir(target_path)
    for img in os.listdir(os.path.join(img_path,"images")):
        if target_name in img:
            shutil.copy(os.path.join(img_path,"images",img),os.path.join(target_path,img))
            img_num = img_num+1
            # print(img)
    print("Copy of {} has been completed!img number is {}".format(target_name,img_num))

#--data_idx ${data_idx} --exp_idx ${exp_idx} --epoch ${i} --phase ${phase} --mask_name ${maskname1} --cam_name ${maskname2}
#maskname1="a2_b"
#maskname2="G_A_A2B_cam"
def fuse_cam_a2b(opt,cam,a2b,fused):
    #path
    cam_path = os.path.join(opt.results_dir,opt.name,opt.phase+"_"+opt.epoch+"_"+cam)
    a2b_path = os.path.join(opt.results_dir,opt.name,opt.phase+"_"+opt.epoch+"_"+a2b)
    result_path = os.path.join(opt.results_dir,opt.name,opt.phase+"_"+opt.epoch+"_"+fused)
    
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

        cam_img = imageio.imread(os.path.join(cam_path,img[:-9]+"_"+cam+".png"))/255
        a2b_img = imageio.imread(os.path.join(a2b_path,img))/255
        if opt.abla_have_cam ==1:
            result_img = cam_img * a2b_img
        elif opt.abla_have_cam ==2:
            result_img = cam_img + a2b_img
        result_img = 1-result_img
        fused_list.append(result_img)
        
        imageio.imsave(os.path.join(result_path,img[:-9]+".png"), (result_img * 255).astype(np.uint8))
    
    print("fused img number is {},all of them have been saved".format(idx+1))
    return fused_list





if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html1.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML


    a2_b = "a2_b"
    cam = "G_A_A2B_cam"
    fused = "fused"
    realA_mask = "realA_mask"

    #移动cam和a2_b到单独的文件夹
    mv_img(opt,a2_b)
    mv_img(opt,realA_mask)
    


    #fused cam and a2_b to get fused masks,and get the list of fused mask
    if opt.abla_have_cam!=0:
        mv_img(opt,cam)
        fused_list = fuse_cam_a2b(opt,cam,a2_b,fused)

    


    