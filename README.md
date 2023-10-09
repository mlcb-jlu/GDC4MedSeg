# GDC4MedSeg

# readme.md

## 1.Create environment
-  Python 3.6 and the following python dependencies.
"""
pip install -r prerequisite.txt
"""


## 2.Data Preparations
Download ISIC Dataset and BraTs Dataset. There are two datasets, namely ISIC and BraTS. Each dataset is divided into two categories: diseased and normal. 
Folder A represents the diseased category, while folder B represents the normal category.

Dataset root/
├── BraTs
│    ├──trainA 
│    ├──trainA_label
│       ├── trainB
│    ├── valA
│        ├── valA_label
│        ├── valA_label
│        ├── testA
│        ├── testA_label
│    └── testB
└── ISIC
     ├── trainA
          ……
     └── testB

## 3.Usage
## Dataset BraTs 

step 1. Train the model: Run the ./scripts/train_brats.sh for training model, view the training results in the checkpoints folder. 

step 2. Validate the model: Run the sh ./scripts/val_brats.sh, view the metric results and visualization results for the validation epoch in the results folder, and select the best generation of model parameters as the model parameters for the test dataset."

step 3. Test the model: Run the sh ./scripts/test_brats.sh and achieve the final results.

""" 

# If you need to train the ISIC dataset, you can replace the original attention_gan_model.py and networks.py in the models folder with attention_gan_model_isic.py and networks_isic.py.

"""
