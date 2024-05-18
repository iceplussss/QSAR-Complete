import os
import sys
import numpy as np
import torch as th
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
from deepchem.metrics.score_function  import pearson_r2_score
from data_tools import QsarDataset
from model import QComp
from train import trainer
from utilities import th2np, binary_accuracy, cross_count
from torch.utils.tensorboard import SummaryWriter


device = th.device('cuda' if th.cuda.is_available() else 'cpu')
dtype = th.float32


################################################## 1. Setting ##################################################

########## data path ##########
idx = 0
dataset_name = "fold_" + str(idx)
dataset_type = "regression"

exp_trainset = pd.read_csv('public_data_results/random_split_data_results/{}/public_admet_data_random_{}_train_set.csv'.format(dataset_name, dataset_name))
exp_testset = pd.read_csv('public_data_results/random_split_data_results/{}/public_admet_data_random_{}_test_set.csv'.format(dataset_name, dataset_name))

count_matrix = cross_count(exp_trainset.drop(columns=['smiles']))

qsar_trainset = pd.read_csv('public_data_results/random_split_data_results/{}/chemprop_multitask_pred/public_admet_data_random_{}_train_set_model_pred.csv'.format(dataset_name, dataset_name))
qsar_testset = pd.read_csv('public_data_results/random_split_data_results/{}/chemprop_multitask_pred/public_admet_data_random_{}_test_set_model_pred.csv'.format(dataset_name, dataset_name))

tasks = np.array(exp_trainset.columns[1:])

exp_trainset = exp_trainset.drop(columns=["smiles"]).to_numpy()
exp_testset = exp_testset.drop(columns=["smiles"]).to_numpy()

qsar_trainset = qsar_trainset.drop(columns=["smiles"]).to_numpy()
qsar_testset = qsar_testset.drop(columns=["smiles"]).to_numpy()


########## training ##########
num_epoches = 10
batch_size = 1000

lr = 1e-3
step_size = 1
gamma = 0.5

########## output frequency ##########
freq_output_loss = 1
freq_output_figure = 500

########## metrics ##########
if dataset_type == 'classification':
    metric_list = [
                #     pearson_r2_score, 
                #    r2_score,
                mean_squared_error, 
                roc_auc_score,
                binary_accuracy,
                ]
    metric_name_list = [
                        # "pearson_r2_score",
                        # "r2_score",
                        "mean_squared_error",
                        "roc_auc_score",
                        "binary_accuracy"
                        ]
else:
    metric_list = [
            pearson_r2_score, 
            r2_score,
            mean_squared_error, 
            ]
    metric_name_list = [
                        "pearson_r2_score",
                        "r2_score",
                        "mean_squared_error",
                        ]


################################################## 2. Data Loading ##################################################
dataset = QsarDataset(exp_trainset, qsar_trainset, exp_testset, qsar_testset, assay_names=tasks)

trainset = dataset.get_torch_dataset('train', normalize=False)
testset  = dataset.get_torch_dataset('test', normalize=False)

train_loader = th.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

mute_mask = th.tensor(count_matrix > 50, dtype=dtype, device=device)

################################################## 3. Model ##################################################
qcomp = QComp(size=dataset.feature_size, diagonal=dataset.global_std, mute_mask=mute_mask)


################################################## 4. Training ##################################################
trainer_obj = trainer(qcomp, lr, step_size, gamma)

tb = SummaryWriter()
tb.log_dir = os.path.join(os.path.join("saved_data", dataset_name), tb.log_dir)
if not os.path.exists(tb.log_dir):
    os.makedirs(tb.log_dir)


for idx_epoch in range(num_epoches):
    for idx,batch in enumerate(train_loader):

        n_iter = idx_epoch * len(train_loader) + idx

        loss = trainer_obj.train_step(batch["exp_data"], batch["qsar_data"])

        if n_iter % freq_output_figure == 0:
            with th.no_grad():

                for metric, metric_name in zip(metric_list, metric_name_list):
                    trainer_obj.impute_dataset(testset, metric, metric_name, tb, True, n_iter)

                th.save(qcomp.state_dict(), os.path.join(tb.log_dir, "model.pth"))

        if n_iter % freq_output_loss == 0:
            with th.no_grad():
                print("-------------------- n_iter: {} --------------------".format(n_iter))
                print("Epoch", idx_epoch, "Batch", idx, "Loss", loss.item(), "Learning Rate", trainer_obj.scheduler.get_lr()[0])
                tb.add_scalar("loss", loss.item(), n_iter)
                tb.add_scalar("learning_rate", trainer_obj.scheduler.get_lr()[0], n_iter)

                eigenv_sqrt = th.linalg.eigvalsh(qcomp.forward()) ** 0.5
                print("eigenv_sqrt of Sigma", eigenv_sqrt)
                tb.add_scalar("eigenv_sqrt max", th2np(eigenv_sqrt)[-1], n_iter)
                tb.add_scalar("eigenv_sqrt min", th2np(eigenv_sqrt)[0], n_iter)

                print("transform_matrix", th2np(qcomp.qsar_trans_M).diagonal())

                print("-----------------------------------------------------------------")

    trainer_obj.scheduler_step()
