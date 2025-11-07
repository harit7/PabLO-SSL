source activate ssl

N_TH=1000
N_CAL=1000
SEED=0
GPU=0
export OPT_TH=YES
export MODEL_SELECTION=Cov

# Fixmatch + Ours
python train.py \
--use_tensorboard \
--c config/classic_cv/fixmatch/fixmatch_cifar10_250_0.yaml \
--save_name determine \
--prefix fixmatch-mr_cifar10 \
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
--save_dir saved_models \
--bayes True \
--bam_config ./config/bam/default.yaml