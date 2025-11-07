N_TH=0
N_CAL=0
SEED=0
GPU=0

# Fixmatch
python train.py \
--c config/classic_cv/fixmatch/fixmatch_cifar100_2500_0.yaml \
--prefix fixmatch_baseline \
--use_post_hoc_calib False \
--n_cal ${N_CAL} \
--n_th ${N_TH} \
--take_d_cal_th_from eval \
--loss_reweight False \
--aug_1 weak \
--aug_2 strong \
--gpu ${GPU} \
--accumulate_pseudo_labels False \
--seed ${SEED} &

# Freematch
python train.py \
--c config/classic_cv/freematch/freematch_cifar100_2500_0.yaml \
--prefix freematch_baseline \
--use_post_hoc_calib False \
--n_cal ${N_CAL} \
--n_th ${N_TH} \
--take_d_cal_th_from eval \
--loss_reweight False \
--aug_1 weak \
--aug_2 strong \
--gpu ${GPU} \
--accumulate_pseudo_labels False \
--seed ${SEED} &

# Fixmatch + BAM
python train.py \
--c config/classic_cv/fixmatch/fixmatch_cifar100_2500_0.yaml \
--prefix fixmatch_bam \
--use_post_hoc_calib False \
--n_cal ${N_CAL} \
--n_th ${N_TH} \
--take_d_cal_th_from eval \
--loss_reweight False \
--aug_1 weak \
--aug_2 strong \
--gpu ${GPU} \
--accumulate_pseudo_labels False \
--bayes True \
--bam_config ./config/bam/default.yaml \
--seed ${SEED} &

# Fixmatch + MR
python train.py \
--c config/classic_cv/fixmatch/fixmatch_cifar100_2500_0.yaml \
--prefix fixmatch_mr \
--use_post_hoc_calib False \
--n_cal ${N_CAL} \
--n_th ${N_TH} \
--take_d_cal_th_from eval \
--loss_reweight False \
--aug_1 weak \
--aug_2 strong \
--gpu ${GPU} \
--accumulate_pseudo_labels False \
--mr True \
--mr_config ./config/mr/default.yaml \
--seed ${SEED} &