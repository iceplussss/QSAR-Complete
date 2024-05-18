import torch as th 
import numpy as np

class QsarDataset:
    def __init__(self, exp_trainset, qsar_trainset, exp_testset, qsar_testset, assay_names=None):
        '''
        Args:
            exp_trainset: numpy array of shape (#data points, #assays)
            qsar_trainset: numpy array of shape (#data points, #assays)
            exp_testset: numpy array of shape (#data points, #assays)
            qsar_testset: numpy array of shape (#data points, #assays)
        '''
        self.exp_trainset = exp_trainset
        self.qsar_trainset = qsar_trainset
        self.exp_testset = exp_testset
        self.qsar_testset = qsar_testset
        self.feature_size = exp_trainset.shape[1]
        self.assay_names = assay_names

    @property
    def global_mean(self):
        return np.nanmean(self.exp_trainset, axis=0)
    
    @property
    def global_std(self):
        return np.nanstd(self.exp_trainset, axis=0)
    
    def get_torch_dataset(self, dataset_type, dtype=th.float32, device='cpu', normalize=True):
        '''
        Args:
            type: 'train' or 'test'
            dtype: torch data type
            device: torch device
            normalize: whether to normalize the data
        Returns:
            A TorchDataset
        '''
        if dataset_type == 'train':
            exp_data = self.exp_trainset
            qsar_data = self.qsar_trainset
        elif dataset_type == 'test':
            exp_data = self.exp_testset
            qsar_data = self.qsar_testset
        else:
            raise ValueError
        
        if normalize:
            exp_data = (exp_data - self.global_mean)/self.global_std
            qsar_data = (qsar_data - self.global_mean)/self.global_std
        exp_data = th.tensor(exp_data, dtype=dtype, device=device)
        qsar_data = th.tensor(qsar_data, dtype=dtype, device=device)

        return TorchDataset(
            exp_data = exp_data, 
            qsar_data = qsar_data,
            assay_names = self.assay_names)


class TorchDataset(th.utils.data.Dataset):
    def __init__(self, exp_data, qsar_data, assay_names=None):
        self.exp_data = exp_data
        self.qsar_data = qsar_data
        self.assay_names = assay_names
    
    def __len__(self):
        return len(self.exp_data)
    
    def __getitem__(self, idx):
        return {
            "exp_data": self.exp_data[idx],
            "qsar_data": self.qsar_data[idx]
        }
    