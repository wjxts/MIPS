from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, r2_score, f1_score
import pandas as pd
import os
import numpy as np
import torch 
import logging 

# r2在OLS的训练集上, 意义就是拟合相关系数的平方; 但一般的拟合不成立

logger = logging.getLogger(__name__)
### Evaluator for graph classification
class Evaluator:
    def __init__(self, name=None, eval_metric=None, n_tasks=None):
        self.name = name
        self.eval_metric = eval_metric
        self.n_tasks = n_tasks
       

    def _parse_and_check_input(self, y_true, y_pred, valid_ids=None):
        # converting to torch.Tensor to numpy on cpu
        if torch is not None and isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if torch is not None and isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        ## check type
        if not isinstance(y_true, np.ndarray):
            raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

        if not y_true.shape == y_pred.shape:
            raise RuntimeError('Shape of y_true and y_pred must be the same')

        if not y_true.ndim == 2:
            raise RuntimeError('y_true and y_pred mush to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

        # if not y_true.shape[1] == self.n_tasks:
        #     raise RuntimeError('Number of tasks for {} should be {} but {} given'.format(self.name, self.n_tasks, y_true.shape[1]))
        if valid_ids is not None and isinstance(valid_ids, torch.Tensor):
            valid_ids = valid_ids.detach().cpu().numpy()
        return y_true, y_pred, valid_ids


    def eval(self, y_true, y_pred, valid_ids=None, mean=None, std=None):
        y_true, y_pred, valid_ids = self._parse_and_check_input(y_true, y_pred, valid_ids)
        if self.eval_metric in ['rmse', 'mae', 'r2']:
            if mean is None or std is None:
                raise ValueError('mean and std should be provided for rmse, mae and r2')
            print(f"eval: {mean}, {std}")
            self.mean = mean 
            self.std = std
        if self.eval_metric == 'rocauc':
            return self._eval_rocauc(y_true, y_pred)
        elif self.eval_metric == 'rocauc_resp':
            return self._eval_rocauc_resp(y_true, y_pred, valid_ids)
        elif self.eval_metric == 'ap':
            return self._eval_ap(y_true, y_pred)
        elif self.eval_metric == 'ap_resp':
            return self._eval_ap_resp(y_true, y_pred)
        elif self.eval_metric == 'rmse':
            return self._eval_rmse(y_true, y_pred)
        elif self.eval_metric == 'acc':
            return self._eval_acc(y_true, y_pred)
        elif self.eval_metric == 'f1':
            return self._eval_f1(y_true, y_pred)
        elif self.eval_metric == 'mae':
            return self._eval_mae(y_true, y_pred)
        elif self.eval_metric == 'r2':
            return self._eval_r2(y_true, y_pred)
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

    
    def _eval_rocauc(self, y_true, y_pred):
        '''
            compute ROC-AUC averaged across tasks
        '''

        rocauc_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:,i] == y_true[:,i]
                rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return sum(rocauc_list)/len(rocauc_list)
    
    def _eval_rocauc_resp(self, y_true, y_pred, valid_ids=None):
        '''
            compute ROC-AUC averaged across tasks
        '''

        rocauc_list = []
        
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:,i] == y_true[:,i]
                if valid_ids is not None:
                    is_labeled = np.logical_and(is_labeled, valid_ids[:, i])
                if len(y_true[is_labeled,i] != 0):
                    rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return rocauc_list

    def _eval_f1(self, y_true, y_pred):
        '''
            compute ROC-AUC averaged across tasks
        '''

        f1_list = []
        y_pred = (y_pred>0.5)
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:,i] == y_true[:,i]
                f1_list.append(f1_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        if len(f1_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute f1.')

        return sum(f1_list)/len(f1_list)

    def _eval_ap(self, y_true, y_pred):
        '''
            compute Average Precision (AP) averaged across tasks
        '''

        ap_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:,i] == y_true[:,i]
                if len(y_true[is_labeled,i] != 0):
                    ap = average_precision_score(y_true[is_labeled,i], y_pred[is_labeled,i])

                    ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')

        return sum(ap_list)/len(ap_list)
    def _eval_ap_resp(self, y_true, y_pred):
        '''
            compute Average Precision (AP) averaged across tasks
        '''

        ap_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:,i] == y_true[:,i]
                ap = average_precision_score(y_true[is_labeled,i], y_pred[is_labeled,i])

                ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')

        return ap_list

    def _eval_rmse(self, y_true, y_pred):
        '''
            compute RMSE score averaged across tasks
        '''
        # print(type(y_true), type(y_pred), type(self.mean), type(self.std)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'torch.Tensor'> <class 'torch.Tensor'>
        # y_true: unscale, y_pred: model's output, which predict the standized output
        rmse_list = []
        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = y_true[:,i] == y_true[:,i]
            if (self.mean is not None) and (self.std is not None):
                rmse_list.append(np.sqrt(((y_true[is_labeled,i] - (y_pred[is_labeled,i]*self.std[i]+self.mean[i]))**2).mean()))
                # rmse_list.append(np.sqrt(((y_true[is_labeled,i] - (y_pred[is_labeled,i]*self.std[i]))**2).mean()))
            else:
                rmse_list.append(np.sqrt(((y_true[is_labeled,i] - y_pred[is_labeled,i])**2).mean()))
        return sum(rmse_list)/len(rmse_list)
    
    def _eval_mae(self, y_true, y_pred):
        '''
            compute MAE score averaged across tasks
        '''
        mae_list = []

        for i in range(y_true.shape[1]):
            # ignore nan values
            # print((y_pred[:,i] != y_pred[:,i]).sum())
            is_labeled = y_true[:,i] == y_true[:,i]
            if (self.mean is not None) and (self.std is not None):
                mae_list.append(mean_absolute_error(y_true[is_labeled,i], y_pred[is_labeled,i]*self.std[i]+self.mean[i]))
            else:
                mae_list.append(mean_absolute_error(y_true[is_labeled,i], y_pred[is_labeled,i]))
            # if (self.mean is not None) and (self.std is not None):
            #     mae_list.append(mean_absolute_error(y_true[:,i], y_pred[:,i]*self.std[i]+self.mean[i]))
            # else:
            #     mae_list.append(mean_absolute_error(y_true[:,i], y_pred[:,i]))

        return sum(mae_list)/len(mae_list)
    def _eval_r2(self, y_true, y_pred):
        '''
            compute R2 score averaged across tasks
        '''
        r2_list = []

        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = y_true[:,i] == y_true[:,i]
            if (self.mean is not None) and (self.std is not None):
                r2_list.append(r2_score(y_true[is_labeled,i], y_pred[is_labeled,i]*self.std[i]+self.mean[i]))
            else:
                r2_list.append(r2_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        # return r2_list
        return sum(r2_list)/len(r2_list)

    def _eval_acc(self, y_true, y_pred):
        acc_list = []
        y_pred = (y_pred>0.5)
        for i in range(y_true.shape[1]):
            is_labeled = y_true[:,i] == y_true[:,i]
            correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
            # print(len(correct))
            acc_list.append(float(np.sum(correct))/len(correct))

        return sum(acc_list)/len(acc_list)