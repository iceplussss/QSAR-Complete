import torch.nn as nn
import torch as th  


class PositiveDefiniteMatrix(nn.Module):
    def __init__(self, size):
        super(PositiveDefiniteMatrix, self).__init__()
        self.L = nn.Parameter(th.randn(size, size))
    
    def forward(self):
        return th.matmul(self.L, self.L.t())
    

class QComp(nn. Module):
    def __init__(self, size, diagonal=None, mute_mask=None):
        super(QComp, self).__init__()
        self.n = size

        ndof = size* (size+1)//2
        initial_lower_triangular = 0.1 * th.randn(ndof)
        if diagonal is None:
            initial_diagonal = th.ones(size, dtype=initial_lower_triangular.dtype, device=initial_lower_triangular.device)
        else:
            initial_diagonal = th.tensor(diagonal, dtype=initial_lower_triangular.dtype, device=initial_lower_triangular.device)
        
        ### set the diagonal elements to be the initial diagonal 
        for i in range(size):
            initial_lower_triangular[i*(i+1)//2+i] = initial_diagonal[i]

        self.lower_triangular = nn.Parameter(initial_lower_triangular)
        self.filter_matrix = th.tril(th.ones((self.n, self.n), dtype=th.bool, device=self.lower_triangular.device))

        ##### linear transformation for qsar data
        # self.qsar_trans_M = nn.Parameter(th.eye(size))
        # self.qsar_trans_b = nn.Parameter(th.zeros(size))
        ##### turn off the linear transformation
        self.register_buffer("qsar_trans_M", th.eye(size))
        self.register_buffer("qsar_trans_b", th.zeros(size))

        if mute_mask is None:
            self.register_buffer("mute_mask", th.ones(size,size))
        else:
            mute_mask_diag = th.diag(mute_mask)
            if mute_mask_diag.min() == 0:
                print('please do not mute the diagonal elements of the sigma matrix.')
            self.register_buffer("mute_mask", mute_mask)

    def transform(self, x):
        return th.matmul(x, self.qsar_trans_M) + self.qsar_trans_b

    def forward(self):  
        L = th.zeros((self.n, self.n), dtype=self.lower_triangular.dtype, device=self.lower_triangular.device)
        L[self.filter_matrix] = self.lower_triangular
        return th.matmul(L, L.t())
    
    def regularize(self):  
        tol = 1e-12
        with th.no_grad():
            sigma_original = self.forward()
            sigma = sigma_original * self.mute_mask
            eigenvalue, eigenvectors = th.linalg.eigh(sigma)
            eigenvalue_pd = th.clip(eigenvalue, min=tol, max=None)
            sigma_reconstruct = eigenvectors @ (th.diag(eigenvalue_pd) @ eigenvectors.T)
            if (eigenvalue_pd-eigenvalue).sum() / eigenvalue.sum() > 0.01:
                print('regularized positve definite sigma matrix is not close enough to the original one.')
            L = th.linalg.cholesky(sigma_reconstruct)
        print('mean|sigma_regularized-sigma| / mean|sigma|=', th.abs(sigma_reconstruct-sigma_original).mean() / th.abs(sigma_original).mean() )
        self.lower_triangular.data = L[self.filter_matrix]
        return
    
    def impute(self, exp_data_masked, qsar_data_original ):
        '''
        Args:
            exp_data_masked: a tensor of shape (#assay)
            qsar_data_original: a tensor of shape (#assay)
        '''
        sigma = self.forward()
        qsar_data = self.transform(qsar_data_original).flatten()

        filter_M = th.isnan(exp_data_masked)     # filter for missing data
        filter_O = th.isfinite(exp_data_masked)  # filter for observed data

        if sum(filter_O) == 0:     # nothing is known
            imputed_data = qsar_data
        elif sum(filter_M) == 0:   # nothing is missing
            imputed_data = exp_data_masked
        else:
            diff_O = exp_data_masked[filter_O] - qsar_data[filter_O]

            sigma_OO = sigma[filter_O, :][:, filter_O]
            sigma_MO = sigma[filter_M, :][:, filter_O]

            ## use cholesky decomposition to compute the inverse of sigma_OO
            u = th.linalg.cholesky(sigma_OO)
            inverse_sigma_OO = th.cholesky_inverse(u)

            imputed_M = th.matmul(sigma_MO, th.matmul(inverse_sigma_OO, diff_O))
            imputed_M += qsar_data[filter_M]

            imputed_data = exp_data_masked.clone()
            imputed_data[filter_M] = imputed_M
        return imputed_data
        