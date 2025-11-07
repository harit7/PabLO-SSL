N_TH=3000
N_CAL=3000
SEED=0
GPU=0
RESUME=False
export OPT_TH=YES

# Fixmatch + Ours
python train.py \
--c config/classic_cv/fixmatch/fixmatch_cifar100_2500_0.yaml \
--prefix fixmatch_ours \
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
-- resume ${RESUME} \

# Freematch + Ours
python train.py \
--c config/classic_cv/freematch/freematch_cifar100_2500_0.yaml \
--prefix freematch_ours \
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
-- resume ${RESUME} \
