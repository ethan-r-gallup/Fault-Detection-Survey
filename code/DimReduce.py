import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import Stats


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
        self.T2thresh = Stats.UCL(t2, t2_thresh)
        
        #calculate the SPE upper control limit
        identity = np.identity(data.shape[1])
        q = np.array([xi.T @ (identity - self.PComponents @ self.PComponents.T) @ xi for xi in data.values])
        self.Qthresh = Stats.UCL(q, q_thresh)
        
    def detect(self, y, visualize=True):
        t2 = np.array([yi.T @ self.PComponents @ (np.diag(self.Plam ** -1)) @ self.PComponents.T @ yi for yi in y.values])
        identity = np.identity(y.shape[1])
        q = np.array([yi.T @ (identity - self.PComponents @ self.PComponents.T) @ yi for yi in y.values])
        t_fault = t2 > self.T2thresh
        q_fault = q >self.Qthresh
        fault = t_fault | q_fault
        if visualize:
            self.visualize()
        return fault, t2, q

    def fit_detect(self, data, exp_var=0.99, t2_thresh='hotel', q_thresh='spe', visualize=True):
        self.fit(data, exp_var, t2_thresh, q_thresh)
        return self.detect(data, visualize)

    @staticmethod
    def number_pcs(lam, var):
        var_exp = 0
        num_pcs = 0
        tot_var = np.sum(lam)
        while var_exp < var:
            var_exp += lam[num_pcs] / tot_var
            num_pcs += 1
        return num_pcs

class DiPCA:
    pass

class CVA:
    pass

class ICA:
    pass