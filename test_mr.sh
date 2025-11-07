N_TH=3000
N_CAL=3000
SEED=0
GPU=0

# Fixmatch + MR
python train.py \
--c config/classic_cv/fixmatch/fixmatch_cifar10_250_0.yaml \
--use_post_hoc_calib False \
--n_cal ${N_CAL} \
--n_th ${N_TH} \
--take_d_cal_th_from eval \
--loss_reweight False \
--aug_1 weak \
--aug_2 strong \
--gpu 0 \
--accumulate_pseudo_labels False \
--mr True \
--mr_config ./config/mr/default.yaml \
--seed ${SEED}