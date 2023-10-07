import os.path
from data.base_dataset import BaseDataset, get_transform, get_transform_semi
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np




class UnalignedvocDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.label_npy_path = opt.label_npy
        self.img_list_path = opt.img_list
        self.label_npy = self.load_image_label_list_from_npy(self.label_npy_path)

        self.semi_train = opt.semi_train

        # import pdb;pdb.set_trace()

        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image

        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

        if self.semi_train == 1:#半监督
            self.dir_semi_A = os.path.join(opt.dataroot, opt.phase + '_semi_A')  #'/path/to/data/train_semi_A'
            self.dir_semi_B = os.path.join(opt.dataroot, opt.phase + '_semi_B')  #'/path/to/data/train_semi_B'
            self.dir_semi_A_label = os.path.join(opt.dataroot,opt.phase + '_semi_A_label')   #'/path/to/data/train_semi_A_label'

            self.semi_A_paths = sorted(make_dataset(self.dir_semi_A, opt.max_dataset_size))
            self.semi_B_paths = sorted(make_dataset(self.dir_semi_B, opt.max_dataset_size))
            self.semi_A_label_paths = sorted(make_dataset(self.dir_semi_A_label, opt.max_dataset_size))

            self.semi_A_size = len(self.semi_A_paths)
            self.semi_B_size = len(self.semi_B_paths)
            self.semi_A_label_size = len(self.semi_A_label_paths)

            seed = np.random.randint(2147483647)
            random.seed(seed) 
            self.transform_semi_A_label = get_transform(self.opt, grayscale=True)
            self.transform_semi_A = get_transform(self.opt, grayscale=(input_nc == 1))
            self.transform_semi_B = get_transform(self.opt, grayscale=(output_nc == 1))



    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        
        A_path = self.A_paths[index % self.A_size]
        
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_filename = A_path.split('/')[-1]
        B_filename = B_path.split('/')[-1]
        A_label = self.label_npy[A_filename[:-4]]
        B_label = self.label_npy[B_filename[:-4]]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        
        if self.semi_train == 1:
            semi_A_path = self.semi_A_paths[index % self.semi_A_size]
            if self.opt.serial_batches:   # make sure index is within then range
                index_semi_B = index % self.semi_B_size
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_semi_B = random.randint(0, self.semi_B_size - 1)
            semi_B_path = self.semi_B_paths[index_semi_B]
            semi_A_lebel_path = self.semi_A_label_paths[index % self.semi_A_label_size]

            semi_A_img = Image.open(semi_A_path).convert('RGB')
            semi_B_img = Image.open(semi_B_path).convert('RGB')
            semi_A_label_img = Image.open(semi_A_lebel_path).convert('RGB')

            semi_A = self.transform_A(semi_A_img)
            semi_B = self.transform_A(semi_B_img)
            semi_A_label = self.transform_A(semi_A_label_img)

            return {'A': A, 'B': B, \
                    'semi_A':semi_A,'semi_B':semi_B,'semi_A_label':semi_A_label,\
                    'A_paths': A_path, 'B_paths': B_path,\
                    "semi_A_paths": semi_A_path,"semi_B_paths": semi_B_path,"semi_A_label_paths": semi_A_lebel_path}
        else: 
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path,'A_label':A_label,'B_label':B_label}

        

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


    # def load_image_label_list_from_npy_useless(img_name_list):

    #     cls_labels_dict = np.load('/home/dw/tangsy/psa-master/voc12/cls_labels_brats.npy', allow_pickle=True).item()
    #     return [cls_labels_dict[img_name] for img_name in img_name_list]

    def load_image_label_list_from_npy(self,path):

        cls_labels_dict = np.load(path, allow_pickle=True).item()
        return cls_labels_dict
