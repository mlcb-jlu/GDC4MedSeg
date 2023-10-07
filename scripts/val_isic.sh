
######################完整的并发val脚本###########################

#使用脚本
#nohup sh ./scripts/GithubUpload/isic/val_isic.sh >./log/n_isic_rn1_d1_i10_val.log 2>&1 &

# watch -n 0.1 nvidia-smi

###########说明############
# input:
#     start:开始的iter
#     end：结束的iter
#     thread：线程的数量
###########说明############

#brats start 120   end 160  
#brats i3 start 70 end 100 加了mask


date
data_species=isic
start=70
end=160
thread=3
for iter in $(seq ${start} `expr ${thread} \* 5` ${end})
do
{
        for i in $(seq $iter 5 `expr \( ${thread} - 1 \) \* 5 + $iter`)
        do
        {
            if [ $i -le $end ] 
            then
                date
                for idx in $(seq 1 1)
                do
                    #ixx需要进行修改
                    exp_idx="n_${data_species}_rn1_d"${idx}"_i10"
                    phase="val"

                    data_idx="${data_species}_old_attentiongan/${data_species}_"${idx}

                    maskname="fused"
                                            
                    python test.py --dataroot ./datasets/${data_idx} --name ${exp_idx} --epoch ${i} --phase ${phase} --num_test 517 \
                                    --model attention_gan_model --dataset_mode unaligned --norm instance  \
                                    --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0  --saveDisk
                    python ./util/mask2crf.py --exp_idx ${exp_idx} --epoch ${i} --phase ${phase} --data_idx ${data_idx} --maskname ${maskname}
                    python ./util/erode.py --exp_idx ${exp_idx} --epoch ${i} --phase ${phase} --data_idx ${data_idx} --data_species ${data_species} --kernel_size 3
                    python ./util/five_metrics_crf_deal.py  --exp_idx ${exp_idx} --epoch ${i} --phase ${phase} --data_idx ${data_idx}
                    
                done
                date
            fi
        } &
            
        done
        wait
    
}
done
date

######################完整的并发val脚本###########################





