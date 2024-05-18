import os
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from utilities import th2np


class trainer:

    def __init__(self, model, lr, step_size, gamma):
        self.model = model

        self.optimizer = th.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def log_likelihood(self, exp_data, qsar_data):

        filter_O = th.isfinite(exp_data) # filter for observed data

        diff_O = exp_data[filter_O] - qsar_data[filter_O]
        k_O = len(diff_O)

        sigma = self.model.forward()
        sigma_OO = sigma[filter_O, :][:, filter_O]

        ## use cholesky decomposition to compute the inverse of sigma_OO
        u = th.linalg.cholesky(sigma_OO)
        inverse_sigma_OO = th.cholesky_inverse(u)

        term1 = -0.5 * th.matmul(diff_O, th.matmul(inverse_sigma_OO, diff_O))
        term2 = -0.5 * k_O * th.log(2 * th.tensor(np.pi))
        term3 = -0.5 * th.log(th.det(sigma_OO))

        return term1 + term2 + term3
    
    def train_step(self, exp_dataset, qsar_dataset):

        self.optimizer.zero_grad()

        #### transform qsar_dataset
        qsar_dataset = self.model.transform(qsar_dataset)

        #### compute loss
        loss = 0
        for (exp_data, qsar_data) in zip(exp_dataset, qsar_dataset):
            loss += -self.log_likelihood(exp_data, qsar_data)
        loss /= exp_dataset.size(dim=0)

        loss.backward()
        self.optimizer.step()
        self.model.regularize()  # new
        return loss
    
    def scheduler_step(self):
        self.scheduler.step()

    def impute_dataset(self, testset, metric, metric_name, tb, flag_fig=False, n_iter=0):

        feature_name_list = testset.assay_names
        impute_score_list = []
        qsar_score_list = []

        for target_idx in np.arange(len(feature_name_list)):
            filter_notna = ~th.isnan(testset.exp_data[:,target_idx])

            test_exp_data = testset.exp_data[filter_notna]
            test_qsar_data = testset.qsar_data[filter_notna]

            test_exp_data_masked = test_exp_data.clone()
            test_exp_data_masked[:, target_idx] = th.nan

            imputed_data_col = []
            for (exp_data_masked, qsar_data_original) in zip(test_exp_data_masked, test_qsar_data):
                imputed_data = self.model.impute(exp_data_masked, qsar_data_original)
                imputed_data_col.append(imputed_data)
            imputed_data_col = th.stack(imputed_data_col)

            exp_target = th2np(test_exp_data[:, target_idx]).flatten()
            imputed_target = th2np(imputed_data_col[:, target_idx]).flatten()
            qsar_target = th2np(test_qsar_data[:, target_idx]).flatten()

            impute_score_list.append(metric(exp_target, imputed_target))
            qsar_score_list.append(metric(exp_target, qsar_target))

        if flag_fig:
            fig, ax = plt.subplots(1,1,figsize=(10,10))
            ax.bar(feature_name_list, impute_score_list, label="imputation", alpha=0.5)
            ax.bar(feature_name_list, qsar_score_list, label="qsar", alpha=0.5)
            ax.set_title(metric_name)
            ax.tick_params(axis='x', rotation=90)
            ax.legend()
            plt.tight_layout()
            fig.savefig(os.path.join(tb.log_dir, "{}.png".format(metric_name)))
            tb.add_figure(metric_name, fig, n_iter)