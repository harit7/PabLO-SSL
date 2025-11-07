source activate ssl

N_TH=3000
N_CAL=3000
SEED=0
GPU=1
export OPT_TH=YES

# Fixmatch + Ours
python train.py \
--use_tensorboard \
--c config/classic_cv/fullysupervised/fullysupervised_cifar100_2500_0.yaml \
--save_name determine \
--prefix may-17 \
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


