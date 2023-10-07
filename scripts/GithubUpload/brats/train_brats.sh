
#新brats完整数据集训练脚本
#命名：- n:new 新数据集
#       - rn:re network
#       - d:data idx
#       - i:第几次实验

#各个损失权重
# w_CAM = self.opt.w_CAM
# w_attention_sup_CAM = self.opt.w_attention_sup_CAM
# w_semi = self.opt.w_semi

#训练脚本命令
#nohup sh ./scripts/GithubUpload/train_brats.sh >./log/n_brats_rn1_d1_i10_train.log 2>&1 &
#watch -n 0.1 nvidia-smi

#第几次交叉实验，同时也是数据集的序号，1-5
exp_time=1

#第几次调参实验，一般是用作多次调参，最终确定最优参数
i_idx=10

#network idx
rn_idx=1

#数据集种类 brats ISIC covid
#brats new   isic  old
data_species=brats
new=new

#niter  总共训练的代数>epoch
#epoch continue_train epoch_count同时配合使用
#epoch 加载的代数  epoch_count 继续训练的第一代的命名

#brats w_CAM 100  niter 160 load_size 256 crop_size 256
w_CAM=100

for d_idx in $(seq 1 ${exp_time})
do
    python ./train.py   --batch_size 4 --name n_${data_species}_rn${rn_idx}_d${d_idx}_i${i_idx} \
                        --dataroot ./datasets/${data_species}_${new}_attentiongan/${data_species}_${d_idx}  \
                        --model attention_gan_model --dataset_mode unaligned --pool_size 50 --no_dropout --norm instance \
                        --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --w_CAM ${w_CAM} --w_attention_sup_CAM 10 --w_semi 10 \
                        --load_size 256 --crop_size 256  \
                        --niter 160 --niter_decay 0 --gpu_ids 0 --display_id 0 --display_freq 100 --print_freq 100 \
                        --first_save_epoch 70 --gpu_id 0 \
                        # --epoch 50 --continue_train --epoch_count 51
done

