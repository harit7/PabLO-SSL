source activate ssl

N_TH=3000
N_CAL=3000
SEED=0
GPU=0
export OPT_TH=YES
export MODEL_SELECTION=Cov

# Fixmatch + Ours
python train.py \
--use_tensorboard \
--c config/classic_cv/fixmatch/fixmatch_cifar100_2500_0.yaml \
--save_name determine \
--prefix fixmatch-ours_cifar100 \
--gpu ${GPU} \
--seed ${SEED} \
--epoch 1000 \
--num_train_iter 25000 \
--aug_1 weak \
--aug_2 strong \
--should_use_cache True \
--accumulate_pseudo_labels True \
--use_post_hoc_calib True \
--post_hoc_name pablo \
--full_pl_flag True \
--batch_pl_flag False \
--full_pl_freq 100 \
--freq_schedule False \
--n_cal ${N_CAL} \
--n_th ${N_TH} \
--take_d_cal_th_from eval \
--use_prev_model False \
--post_hoc_eps 0.05 \
--post_hoc_l2 100.0 \
--post_hoc_bs 64 \
--post_hoc_lr 0.01 \
--post_hoc_wd  0.01
