N_TH=3000
N_CAL=3000
SEED=0
GPU=1
export OPT_TH=YES

# Fixmatch + Ours
python train.py \
--c config/classic_cv/fixmatch/fixmatch_cifar10_250_0.yaml \
--use_post_hoc_calib True \
--n_cal ${N_CAL} \
--n_th ${N_TH} \
--take_d_cal_th_from eval \
--loss_reweight False \
--aug_1 weak \
--aug_2 strong \
--gpu ${GPU} \
--accumulate_pseudo_labels True \
--seed ${SEED} \
--full_pl_flag True \
--batch_pl_flag False \
--use_cache True &

# Fixmatch + Ours + Cache
python train.py \
--c config/classic_cv/fixmatch/fixmatch_cifar10_250_0.yaml \
--use_post_hoc_calib True \
--n_cal ${N_CAL} \
--n_th ${N_TH} \
--take_d_cal_th_from eval \
--loss_reweight False \
--aug_1 weak \
--aug_2 strong \
--gpu ${GPU} \
--accumulate_pseudo_labels True \
--seed ${SEED} \
--full_pl_flag True \
--batch_pl_flag False \
--use_cache False &