


#################完整的test 脚本#################################
i=85
crf_t=8

data_idx=1
exp_idx=10
data_species="isic"




exp_idx="n_"${data_species}"_rn1_d"${data_idx}"_i"${exp_idx}

phase="test"
data_idx=${data_species}"_old_attentiongan/"${data_species}"_"${data_idx}

#a2_b G_A_A2B_cam fused
# maskname="a2_b"
# maskname="G_A_A2B_cam"
maskname="fused"

#########################################################################
# abla_have_cam 0 没有cam，不生成fused文件夹,不对空的cam文件夹进行操作
# abla_have_cam 1 融合方式一，相乘，不设置的时候默认为1
# abla_have_cam 2 融合方式二，相加

# a2_b时，crf.py文件需要加上suffix，在进行crf文件读取是加上文件的后缀
# fused时，文件命名是没有后缀的,不需要加上suffix
#########################################################################

python test.py --dataroot ./datasets/${data_idx} --name ${exp_idx} --epoch ${i} --phase ${phase} --num_test 90 \
                                    --model attention_gan_model --dataset_mode unaligned --norm instance  \
                                    --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0  --saveDisk \
                                    --abla_have_cam 1
python ./util/mask2crf.py --crf_t ${crf_t} --exp_idx ${exp_idx} --epoch ${i} --phase ${phase} --data_idx ${data_idx} --maskname ${maskname} \
                            # --suffix True
python ./util/erode.py --crf_t ${crf_t} --exp_idx ${exp_idx} --epoch ${i} --phase ${phase} --data_idx ${data_idx} --data_species ${data_species} --kernel_size 3 #--bool_erode 0
python ./util/five_metrics_crf_deal.py  --crf_t ${crf_t} --exp_idx ${exp_idx} --epoch ${i} --phase ${phase} --data_idx ${data_idx}
#################完整的test 脚本#################################