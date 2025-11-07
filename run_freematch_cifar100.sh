source activate ssl

N_TH=3000
N_CAL=3000
SEED=0
GPU=0
export OPT_TH=YES

# Freematch + Ours
python train.py \
--use_tensorboard \
--c config/classic_cv/freematch/freematch_cifar100_2500_0.yaml \
--save_name determine \
--prefix May-19 \
--gpu ${GPU} \
--seed ${SEED} \
--epoch 1000 \
--num_train_iter 25000 \
--aug_1 weak \
--aug_2 strong \
--should_use_cache False \
--accumulate_pseudo_labels False \
--use_post_hoc_calib False \
--full_pl_flag False \
--batch_pl_flag True \
--full_pl_freq 100 \
--freq_schedule False \