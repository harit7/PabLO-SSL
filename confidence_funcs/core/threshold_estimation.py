import os
from time import time

import numpy as np
import torch
import torch.multiprocessing as mp
from sklearn.metrics import accuracy_score


def old(
    lst_classes, inf_out, auto_lbl_conf, val_ds, val_idcs, logger, err_threshold=0.01
):
    # run inference on the validation points.
    # inference will give, label + conf for each poitn
    # for each conf find the accuracy on points with conf >= this conf
    # settle for the lowest conf that satisifies the accuracy requirement
    # if no conf found, return t_max
    max_t = float("inf")

    Y_val = val_ds.Y.cpu().numpy()
    y_hat = inf_out["labels"].cpu().numpy()
    n_v = len(y_hat)

    score_type = auto_lbl_conf["score_type"]

    scores = inf_out[score_type]

    fast = False

    if fast and len(scores) > 500:
        # print('here')
        min_score = min(scores)
        max_score = max(scores)
        delta = (max_score - min_score) / 20000

        logger.debug(
            "MAX score = {}, MIN score = {}, delta = {}".format(
                max_score, min_score, delta
            )
        )

        if abs(max_score - min_score) <= 1e-8:
            lst_th = [min_score]
        else:
            lst_th = np.arange(min_score, max_score + delta, delta)
    else:
        lst_th = scores

    lst_th = np.array(lst_th)

    S = np.zeros((n_v, 8))

    S[:, 0] = np.arange(0, n_v, 1)
    S[:, 1] = Y_val
    S[:, 2] = y_hat
    S[:, 3] = S[:, 1] == S[:, 2]
    S[:, 4] = inf_out["confidence"]
    # S[:,5] = inf_out['logits']
    # S[:, 6] = inf_out["abs_logit"]

    # if "energy_score" in inf_out:
    #     S[:, 7] = inf_out["energy_score"]

    scores_id_map = {"confidence": 4, "logits": 5, "abs_logit": 6, "energy_score": 7}

    score_key = scores_id_map[score_type]

    # sort in descending order of score
    S = S[(-S[:, score_key]).argsort()]

    def get_err_at_th(S_y, c):
        S2 = S_y[S_y[:, score_key] >= c]
        if len(S2) > 0:
            return 1 - (S2[:, 3].sum() / len(S2))
        else:
            return 1.0

    def get_std_at_th(S_y, c):
        S2 = S_y[S_y[:, score_key] >= c]
        if len(S2) > 0:
            # print(S2.shape)
            # print(S2[:,3].shape)
            z = np.std(1 - (S2[:, 3]))
            # print(z)
            return np.std(1 - (S2[:, 3]))
        else:
            return 0

    C_1 = auto_lbl_conf["C_1"]
    ucb = auto_lbl_conf["ucb"]

    logger.debug(f"C_1 = {C_1} UCB = {ucb}")

    def get_threshold(S_y):

        # std_th = np.array(std_th)
        # print(err_th)
        n_v_t = np.array([len(S_y[S_y[:, score_key] >= th]) for th in lst_th])

        n_v_0 = 10

        lst_th_ = lst_th[np.where(n_v_t > n_v_0)]
        n_v_t_ = n_v_t[np.where(n_v_t > n_v_0)]
        err_th = [get_err_at_th(S_y, th) for th in lst_th_]

        # std_th = [get_std_at_th(S_y,th) for th in lst_th]

        err_th = np.array(err_th)

        # n_v_t = np.array([len(S_y[S_y[:,score_key]>=th]) for th in lst_th]) +10
        # err_th = err_th + C_1*np.sqrt(1/n_v_t)
        if ucb == "hoeffding":
            err_th = err_th + C_1 * np.sqrt(1 / n_v_t_)

        elif ucb == "sigma":
            err_th = err_th + C_1 * np.sqrt(err_th * (1 - err_th))

        # err_th = err_th + C_1*np.sqrt(err_th*(1-err_th))
        # err_th =  err_th + 2*std_th

        good_th = lst_th_[np.where(err_th <= err_threshold)]

        print(good_th)
        exit()

        if len(good_th) > 0:
            t_y = np.min(good_th)
        else:
            t_y = max_t

        return t_y

    val_idcs_to_rm = []
    lst_t_y = []
    class_wise = auto_lbl_conf["class_wise"]

    if class_wise == "independent":
        for y in lst_classes:
            S_y = S[S[:, 2] == y]
            # print(len(S_y),S_y[0])
            t_y = get_threshold(S_y)
            lst_t_y.append(t_y)
            logger.debug(
                "auto-labeling threshold t_i={} for class {}   ".format(t_y, y)
            )

            if t_y < max_t:
                idcs_vals_rm = [
                    val_idcs[i]
                    for i in range(n_v)
                    if y_hat[i] == y and scores[i] >= t_y
                ]
                val_idcs_to_rm.extend(idcs_vals_rm)

    elif class_wise == "joint":
        t_ = get_threshold(S)
        lst_t_y = [t_] * (len(lst_classes))

        logger.debug("auto-labeling threshold t={} for each class.  ".format(t_))

        if t_ < max_t:
            val_idcs_to_rm = [val_idcs[i] for i in range(n_v) if scores[i] >= t_]

    cov = len(val_idcs_to_rm) / len(val_idcs)

    logger.debug(f"coverage while threshold estimation : {cov}")

    return lst_t_y, val_idcs_to_rm, val_err, cov


def new_joint(
    lst_classes, inf_out, auto_lbl_conf, val_ds, val_idcs, logger, err_threshold=0.01
):

    y_val = val_ds.Y.cuda()
    y_hat = inf_out["labels"].cuda()
    val_idcs = torch.tensor(val_idcs).cuda()
    N_v = len(y_hat)
    score_type = auto_lbl_conf["score_type"]
    scores = torch.tensor(inf_out[score_type]).cuda()
    C_1 = auto_lbl_conf["C_1"]

    list_of_thresholds = scores
    where_matrix = scores[:, None] >= list_of_thresholds
    N_t = where_matrix.sum(dim=0)
    N_e = ((y_val[:, None] != y_hat[:, None]) & where_matrix).sum(dim=0)
    err_th = N_e / N_t
    err_th += C_1 * torch.sqrt(err_th * (1 - err_th))
    good_th = list_of_thresholds[err_th < err_threshold]

    if len(good_th) > 0:
        t_y = torch.min(good_th).item()
    else:
        t_y = torch.tensor(float("inf")).cuda().item()

    lst_t_y = [t_y] * len(lst_classes)
    val_idcs_to_rm = []
    if t_y < float("inf"):
        val_idcs_to_rm = val_idcs[scores >= t_y]
    cov = len(val_idcs_to_rm) / len(val_idcs)
    return lst_t_y, val_idcs_to_rm, None, cov


def new(
    lst_classes, inf_out, auto_lbl_conf, val_ds, val_idcs, logger, err_threshold=0.01
):
    if auto_lbl_conf["class_wise"] == "joint":
        return new_joint(
            lst_classes, inf_out, auto_lbl_conf, val_ds, val_idcs, logger, err_threshold
        )

    val_ds.Y = val_ds.Y.cuda()
    # print(val_ds.Y.device, inf_out["labels"].device)

    # err_rate = 1 - torch.sum(val_ds.Y == inf_out["labels"]) / len(
    #    val_ds.Y
    # )  # accuracy_score(val_ds.Y.numpy(), inf_out["labels"])
    err_rate = 0

    val_idcs = torch.tensor(val_idcs).to("cuda")

    classes = lst_classes
    y_true = val_ds.Y.to("cuda")
    y_pred = inf_out["labels"].to("cuda")
    C_1 = torch.tensor(auto_lbl_conf["C_1"]).to("cuda")
    err_threshold = torch.tensor(err_threshold).to("cuda")

    class_to_idx = {class_: y_pred == class_ for class_ in classes}
    scores = torch.tensor(
        inf_out[auto_lbl_conf.score_type], dtype=torch.float32, device="cuda"
    )

    n_v_0 = 10
    # val_idcs_to_rm = [torch.tensor([]).long().to("cuda") for _ in range(len(classes))]

    thresholds = [None for _ in range(len(classes))]

    N_a_v = 0
    for i, class_ in enumerate(classes):
        N_t_class = (scores[class_to_idx[class_], None] >= scores).sum(dim=0)
        scores_selected = scores[N_t_class > n_v_0]

        mask = scores[class_to_idx[class_]][:, None] >= scores_selected
        mask_sum = torch.sum(mask, dim=0)

        correct_predictions_sum = (
            scores[class_to_idx[class_] & (y_pred == y_true)][:, None]
            >= scores_selected
        ).sum(dim=0)

        err_class = torch.where(
            mask_sum > 0, 1 - correct_predictions_sum / mask_sum, torch.tensor(1.0)
        )
        err_class = err_class + C_1 * torch.sqrt(err_class * (1 - err_class))
        candidates = scores_selected[err_class <= err_threshold]

        threshold = (
            torch.min(candidates)
            if candidates.numel() > 0
            else torch.tensor(float("inf"))
        )
        thresholds[i] = threshold.item()

        if torch.isfinite(threshold):
            ll = len(
                val_idcs[class_to_idx[class_]][
                    scores[class_to_idx[class_]] >= threshold
                ]
            )
            # print(ll)
            N_a_v += ll
    logger.info(f"cov : {N_a_v}/{len(scores)}")
    logger.info(f"thresholds : {thresholds}")

    # val_idcs_to_rm = torch.cat(val_idcs_to_rm, dim=0).long().to("cuda")
    cov = N_a_v / len(scores)

    return thresholds, [], err_rate, cov


def per_class(
    scores,
    class_to_idx,
    class_,
    n_v_0,
    y_pred,
    y_true,
    C_1,
    err_threshold,
    val_idcs,
    queue,
    i,
):
    N_t_class = (scores[class_to_idx[class_], None] >= scores).sum(axis=0)
    scores_selected = scores[N_t_class > n_v_0]

    mask = scores[class_to_idx[class_]][:, None] >= scores_selected
    mask_sum = np.sum(mask, axis=0)

    correct_predictions_sum = (
        scores[class_to_idx[class_] & (y_pred == y_true)][:, None] >= scores_selected
    ).sum(axis=0)

    err_class = np.where(
        mask_sum > 0, 1 - correct_predictions_sum / mask_sum, np.array(1.0)
    )
    err_class = err_class + C_1 * np.sqrt(err_class * (1 - err_class))
    candidates = scores_selected[err_class <= err_threshold]

    threshold = np.min(candidates) if candidates.size > 0 else float("inf")

    ll = 0
    if np.isfinite(threshold):
        ll = len(
            val_idcs[class_to_idx[class_]][scores[class_to_idx[class_]] >= threshold]
        )

    queue.put((i, ll, threshold))


def new_parallel(
    lst_classes, inf_out, auto_lbl_conf, val_ds, val_idcs, logger, err_threshold=0.01
):

    val_ds.Y = val_ds.Y.cuda()
    # print(val_ds.Y.device, inf_out["labels"].device)

    # err_rate = 1 - torch.sum(val_ds.Y == inf_out["labels"]) / len(
    #    val_ds.Y
    # )  # accuracy_score(val_ds.Y.numpy(), inf_out["labels"])
    err_rate = 0

    val_idcs = torch.tensor(val_idcs).to("cpu").numpy()

    classes = lst_classes
    y_true = val_ds.Y.to("cpu").numpy()
    y_pred = inf_out["labels"].to("cpu").numpy()
    C_1 = torch.tensor(auto_lbl_conf["C_1"]).to("cpu").numpy()
    err_threshold = torch.tensor(err_threshold).to("cpu").numpy()

    class_to_idx = {class_: y_pred == class_ for class_ in classes}
    scores = torch.tensor(
        inf_out[auto_lbl_conf.score_type], dtype=torch.float32, device="cpu"
    ).numpy()

    n_v_0 = 10
    # val_idcs_to_rm = [torch.tensor([]).long().to("cuda") for _ in range(len(classes))]

    thresholds = [None for _ in range(len(classes))]
    N_a_v = 0

    queue = mp.Queue()
    processes = []

    for i, class_ in enumerate(classes):
        p = mp.Process(
            target=per_class,
            args=(
                scores,
                class_to_idx,
                class_,
                n_v_0,
                y_pred,
                y_true,
                C_1,
                err_threshold,
                val_idcs,
                queue,
                i,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    queue = [queue.get() for _ in range(len(classes))]
    N_a_v = sum(map(lambda x: x[1], queue))
    cov = N_a_v / len(scores)

    ordered_queue = sorted(queue, key=lambda x: x[0])
    thresholds = [x[2] for x in ordered_queue]

    return thresholds, [], err_rate, cov


def determine_threshold(
    lst_classes, inf_out, auto_lbl_conf, val_ds, val_idcs, logger, it, err_threshold=0.01
):

    if err_threshold == "schedule-1":
        if it >= 0 and it < 5000:
            err_threshold = 0.4
        elif it >= 5000 and it < 10000:
            err_threshold = 0.2
        elif it >= 10000 and it < 15000:
            err_threshold = 0.1
        elif it >= 15000 and it < 20000:
            err_threshold = 0.05
        else:
            err_threshold = 0.01
    elif err_threshold == "schedule-2":
        if it >= 0 and it < 5000:
            err_threshold = 0.01
        elif it >= 5000 and it < 10000:
            err_threshold = 0.05
        elif it >= 10000 and it < 15000:
            err_threshold = 0.1
        elif it >= 15000 and it < 20000:
            err_threshold = 0.2
        else:
            err_threshold = 0.4
    else:
        err_threshold = float(err_threshold)
    if os.environ["OPT_TH"] == "YES":
        # print(
        #    "environment variable OPT_TH is set to YES, using optimized threshold estimation"
        # )
        tic = time()
        ret = new(
            lst_classes, inf_out, auto_lbl_conf, val_ds, val_idcs, logger, err_threshold
        )
        toc = time()
        time_elapsed = toc - tic
        # print("time taken for optimized threshold estimation: ", time_elapsed)
    elif os.environ["OPT_TH"] == "NO":
        print(
            "environment variable OPT_TH is set to NO, using non-optimized threshold estimation"
        )
        tic = time()
        ret = old(
            lst_classes, inf_out, auto_lbl_conf, val_ds, val_idcs, logger, err_threshold
        )
        toc = time()
        time_elapsed = toc - tic
        print("time taken for non-optimized threshold estimation: ", time_elapsed)

    elif os.environ["OPT_TH"] == "PARALLEL":
        print(
            "environment variable OPT_TH is set to PARALLEL, using parallelized threshold estimation"
        )
        tic = time()
        ret = new_parallel(
            lst_classes, inf_out, auto_lbl_conf, val_ds, val_idcs, logger, err_threshold
        )
        toc = time()
        time_elapsed = toc - tic
        print("time taken for parallelized threshold estimation: ", time_elapsed)
    elif os.environ["OPT_TH"] == "TEST_PARALLEL":
        print(
            "environment variable OPT_TH is set to TEST_PARALLEL, testing if the paralleled version is the same as paralleled version"
        )
        tic = time()
        ret = new_parallel(
            lst_classes, inf_out, auto_lbl_conf, val_ds, val_idcs, logger, err_threshold
        )
        toc = time()
        time_elapsed = toc - tic
        print("time taken for paralleled threshold estimation: ", time_elapsed)

        tic = time()
        old_ret = new(
            lst_classes, inf_out, auto_lbl_conf, val_ds, val_idcs, logger, err_threshold
        )
        toc = time()
        time_elapsed = toc - tic
        print("time taken for non-paralleled threshold estimation: ", time_elapsed)

        assert torch.allclose(torch.tensor(ret[0]), torch.tensor(old_ret[0]))
        assert torch.allclose(torch.tensor(ret[1]), torch.tensor(old_ret[1]))
        assert ret[2] == old_ret[2]
        assert ret[3] == old_ret[3]
    elif os.environ["OPT_TH"] == "TEST":
        print(
            "environment variable OPT_TH is set to TEST, testing if the optimized version is the same as non-optimized version"
        )
        tic = time()
        ret = new(
            lst_classes, inf_out, auto_lbl_conf, val_ds, val_idcs, logger, err_threshold
        )
        toc = time()
        time_elapsed = toc - tic
        print("time taken for optimized threshold estimation: ", time_elapsed)

        tic = time()
        old_ret = old(
            lst_classes, inf_out, auto_lbl_conf, val_ds, val_idcs, logger, err_threshold
        )
        toc = time()
        time_elapsed = toc - tic
        print("time taken for non-optimized threshold estimation: ", time_elapsed)

        assert torch.allclose(torch.tensor(ret[0]), torch.tensor(old_ret[0]))
        assert torch.allclose(torch.tensor(ret[1]), torch.tensor(old_ret[1]))
        assert ret[2] == old_ret[2]
        assert ret[3] == old_ret[3]
    else:
        ret = new(
            lst_classes, inf_out, auto_lbl_conf, val_ds, val_idcs, logger, err_threshold
        )
    # print()
    return ret


def main():
    import pickle
    import sys

    sys.path.append("../..")
    import semilearn

    with open("args.pkl", "rb") as f:
        args = pickle.load(f)
    old(*args)


if __name__ == "__main__":
    main()
