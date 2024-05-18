import numpy as np
import torch as th 
import scipy.stats as stats 
from sklearn.metrics import r2_score


def th2np(th_tensor):
    return th_tensor.detach().cpu().numpy()

def binary_accuracy(y_true, y_pred):
    class_pred = (y_pred > 0.5).astype(int)
    agreement =   y_true.astype(int) == class_pred
    accuracy = agreement.astype(int).sum() / agreement.size
    return accuracy

def cross_count(df):
    n = df.shape[1]
    count = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(n):
            coexist_filter = df.iloc[:,i].notna() & df.iloc[:,j].notna()
            count[i,j] = coexist_filter.astype(int).sum()
    return count


def performance(exp_dataset, qsar_dataset, target_index, model):

    exp_dataset_masked = exp_dataset.clone()
    exp_dataset_masked[:, target_index] = th.nan

    imputed_dataset = []
    for (exp_data_masked, qsar_data_original) in zip(exp_dataset_masked, qsar_dataset):
        imputed_data = model.impute(exp_data_masked, qsar_data_original)
        imputed_dataset.append(imputed_data)
    imputed_dataset = th.stack(imputed_dataset)

    exp_target = exp_dataset[:, target_index]
    imputed_target = imputed_dataset[:, target_index]
    qsar_target = qsar_dataset[:, target_index]

    ################################# compute scores #################################
    rmse_imputation = np.sqrt(np.mean((th2np(exp_target) - th2np(imputed_target))**2))
    rmse_qsar = np.sqrt(np.mean((th2np(exp_target) - th2np(qsar_target))**2))
    performance_dict = {
        "rmse_imputation": rmse_imputation,
        "rmse_qsar": rmse_qsar
    }

    pearsonr2_imputation = (stats.pearsonr(th2np(exp_target), th2np(imputed_target))[0]) ** 2
    pearsonr2_qsar = (stats.pearsonr(th2np(exp_target), th2np(qsar_target))[0]) ** 2
    performance_dict.update({
        "pearsonr2_imputation": pearsonr2_imputation,
        "pearsonr2_qsar": pearsonr2_qsar
    })

    r2_score_imputation = r2_score(th2np(exp_target), th2np(imputed_target))
    r2_score_qsar = r2_score(th2np(exp_target), th2np(qsar_target))
    performance_dict.update({
        "r2_score_imputation": r2_score_imputation,
        "r2_score_qsar": r2_score_qsar
    })

    class_imputation = (th2np(imputed_target) > 0.5).astype(int)
    agreement_imputation =  th2np(exp_target).astype(int) == class_imputation
    accuracy_imputation = agreement_imputation.astype(int).sum() / agreement_imputation.size
    class_qsar = (th2np(qsar_target) > 0.5).astype(int)
    agreement_qsar =  th2np(exp_target).astype(int) == class_qsar
    accuracy_qsar = agreement_qsar.astype(int).sum() / agreement_qsar.size
    performance_dict.update({
        "accuracy_imputation": accuracy_imputation,
        "accuracy_qsar": accuracy_qsar
    })

    return performance_dict

