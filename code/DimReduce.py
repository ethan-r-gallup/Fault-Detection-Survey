import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, components=None, normalize=True):
        self.n_pcs = None
        self.PComponents = None
        self.RComponents = None
        self.Plam = None
        self.Rlam = None
        self.T2thresh = None
        self.Qthresh = None

    def fit(self, data, exp_var=0.99, t2_thresh='hotel', q_thresh='spe'):
        cov = (1 / (data.shape[0] - 1) * data.T @ data)  # compute the covariance of the data
        eigenvalues, eigenvectors = np.linalg.eig(cov)  # get the eigenvalues and eigenvectors of the covariance matrix
        self.n_pcs = self.number_pcs(eigenvalues, 0.99)  # get number of pcs needed for desired explained variance

        # assign egenvectors and eigenvalues to the principle and residual spaces
        self.PComponents = eigenvectors[:, 0:self.n_pcs]
        self.Plam = eigenvalues[0:self.n_pcs]
        self.RComponents = eigenvectors[:, self.n_pcs:]
        self.Rlam = eigenvalues[self.n_pcs:]

        #calculate Hotellings T^2 upper control limit
        t2 = np.array([xi.T @ self.PComponents @ (np.diag(self.Plam ** -1)) @ self.PComponents.T @ xi for xi in data.values])
        self.T2thresh = stats.UCL(t2, t2_thresh)
        
        #calculate the SPE upper control limit
        identity = np.identity(data.shape[1])
        q = np.array([xi.T @ (identity - self.PComponents @ self.PComponents.T) @ xi for xi in data.values])
        self.Qthresh = stats.UCL(q, q_thresh)
        
    def score(self, y):
        pass

    def fit_score(self, data):
        pass

    @staticmethod
    def number_pcs(lam, var):
        var_exp = 0
        num_pcs = 0
        tot_var = np.sum(lam)
        while var_exp < var:
            var_exp += lam[num_pcs] / tot_var
            num_pcs += 1
        return num_pcs
