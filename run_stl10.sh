## vanilla
#python train.py --c config/classic_cv/fixmatch/fixmatch_stl10_5000_0.yaml --use_post_hoc_calib False --n_cal 2400 --n_th 2400 --take_d_cal_th_from eval --loss_reweight False --aug_1 weak --aug_2 strong --gpu 0 --accumulate_pseudo_labels True
## with PLO
#python train.py --c config/classic_cv/fixmatch/fixmatch_stl10_5000_0.yaml --use_post_hoc_calib True --n_cal 2400 --n_th 2400 --take_d_cal_th_from eval --loss_reweight False --aug_1 weak --aug_2 strong --gpu 0 --accumulate_pseudo_labels True
## vanilla
# python train.py --c config/classic_cv/fixmatch/fixmatch_stl10_1000_0.yaml --use_post_hoc_calib False --n_cal 2400 --n_th 2400 --take_d_cal_th_from eval --loss_reweight False --aug_1 weak --aug_2 strong --gpu 0 --accumulate_pseudo_labels True
## with PLO
# python train.py --c config/classic_cv/fixmatch/fixmatch_stl10_1000_0.yaml --use_post_hoc_calib True --n_cal 2400 --n_th 2400 --take_d_cal_th_from eval --loss_reweight False --aug_1 weak --aug_2 strong --gpu 0 --accumulate_pseudo_labels True

# source activate ssl

N_TH=1000
N_CAL=1000
SEED=0
GPU=1
export OPT_TH=YES

## Fixmatch
python train.py \
--save_name determine \
--c config/classic_cv/fixmatch/fixmatch_stl10_5000_0.yaml \
--use_tensorboard \
--epoch 1000 \
--num_train_iter 25000 \
--prefix stl10_eval_nl250 \
--use_post_hoc_calib False \
--n_cal ${N_CAL} \
--n_th ${N_TH} \
--take_d_cal_th_from eval \
--loss_reweight False \
--aug_1 weak \
--aug_2 strong \
--gpu ${GPU} \
--accumulate_pseudo_labels False \
--seed ${SEED} \
--full_pl_freq 100

GPU=5
# Fixmatch + MR
python train.py \
--save_name determine \
--c config/classic_cv/fixmatch/fixmatch_stl10_5000_0.yaml \
--use_tensorboard \
--epoch 1000 \
--num_train_iter 25000 \
--prefix stl10_eval_nl250 \
--use_post_hoc_calib False \
--n_cal ${N_CAL} \
--n_th ${N_TH} \
--take_d_cal_th_from eval \
--loss_reweight False \
--aug_1 weak \
--aug_2 strong \
--gpu ${GPU} \
--accumulate_pseudo_labels False \
--seed ${SEED} \
--full_pl_flag False \
--batch_pl_flag True \
--full_pl_freq 100 \
--should_use_cache True \
--mr True \
--mr_config ./config/mr/default.yaml &

GPU=4
# # Fixmatch + BAM
python train.py \
--save_name determine \
--c config/classic_cv/fixmatch/fixmatch_stl10_5000_0.yaml \
--use_tensorboard \
--epoch 1000 \
--num_train_iter 25000 \
--prefix stl10_eval_nl250 \
--use_post_hoc_calib False \
--n_cal ${N_CAL} \
--n_th ${N_TH} \
--take_d_cal_th_from eval \
--loss_reweight False \
--aug_1 weak \
--aug_2 strong \
--gpu ${GPU} \
--accumulate_pseudo_labels False \
--seed ${SEED} \
--full_pl_flag False \
--batch_pl_flag True \
--full_pl_freq 100 \
--should_use_cache True \
--bayes True \
--bam_config ./config/bam/default.yaml &

GPU=0
# Fixmatch + Ours
python train.py \
--prefix logits_concat \
--c config/classic_cv/fixmatch/fixmatch_stl10_5000_0.yaml \
--use_tensorboard \
--epoch 1000 \
--num_train_iter 25000 \
--prefix stl10_eval_nl250 \
--use_post_hoc_calib True \
--n_cal ${N_CAL} \
--n_th ${N_TH} \
--take_d_cal_th_from eval \
--loss_reweight False \
--aug_1 weak \
--aug_2 strong \
--gpu ${GPU} \
--accumulate_pseudo_labels False \
--seed ${SEED} \
--full_pl_flag True \
--batch_pl_flag False \
--full_pl_freq 100 \
--should_use_cache True
