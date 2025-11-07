# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/evaluation.py

import os
import pickle
from .hook import Hook


class EvaluationHook(Hook):
    """
    Evaluation Hook for validation during training
    """
    
    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            algorithm.print_fn("validating...")
            eval_dict = algorithm.evaluate('eval')
            algorithm.log_dict.update(eval_dict)
            test_dict = algorithm.evaluate('test')
            algorithm.log_dict.update(test_dict)
            # results_dict['test/best_acc'] = test_dict['test/top-1-acc']

            # update best metrics
            if algorithm.log_dict['eval/top-1-acc'] > algorithm.best_eval_acc:
                algorithm.best_eval_acc = algorithm.log_dict['eval/top-1-acc']
                algorithm.best_it = algorithm.it
    
    def after_run(self, algorithm):
        
        if not algorithm.args.multiprocessing_distributed or (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
            algorithm.save_model('latest_model.pth', save_path)

        results_dict = {'eval/best_acc': algorithm.best_eval_acc, 'eval/best_it': algorithm.best_it}
        if 'test' in algorithm.loader_dict:
            # load the best model and evaluate on test dataset
            best_model_path = os.path.join(algorithm.args.save_dir, algorithm.args.save_name, 'model_best.pth')
            algorithm.load_model(best_model_path)
            test_dict = algorithm.evaluate('test')
            results_dict['test/top1_best_acc'] = test_dict['test/top-1-acc']
            results_dict['test/top5_best_acc'] = test_dict['test/top-5-acc']
        algorithm.results_dict = results_dict
        # Dump the results_dict to a pickle file
        with open(f'{save_path}/results.pkl', 'wb') as f:  
            pickle.dump(algorithm.results_dict, f)
        