import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA


class ICA:
    def __init__(self, components=None):
        if isinstance(components, list):
            self.Whiten = components[0]
            self.Mixing = components[4]
            self.n_components = components[2]
            self.X_new = components[3]
            self.B = components[1]
            self.ica = components[5]
            self.Cov = components[6]

        else:
            self.Whiten = None
            self.Mixing = None
            self.X_new = None
            self.B = None
            self.Cov = None
            self.ica = FastICA(max_iter=100000, whiten=False, tol=0.001)
            self.n_components = self.ica.n_components

    def fit(self, data):
        cov = (1 / (data.shape[0] - 1) * data.T @ data)  # compute the covariance of the data
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        whiten = np.diag(eigenvalues ** -0.5) @ eigenvectors.T
        z = whiten @ data.T

        self.ica = FastICA(max_iter=150000, whiten=False, tol=0.001)
        self.Whiten = whiten
        self.X_new = self.ica.fit_transform(z.T)
        self.n_components = data.shape[1]
        self.Mixing = self.ica.components_
        self.B = (self.Mixing @ (self.Whiten ** -1)).T
        self.Cov = cov
        return [self.Whiten, self.Mixing, self.n_components, self.X_new, self.B, self.ica, self.Cov]

    def score(self, y):
        z = self.Whiten @ y.T
        y_new = self.ica.transform(z.T)
        f_new = np.array([self.Mixing @ yi for yi in y_new])
        i2 = np.array([fi.T @ fi for fi in f_new])

        y_hat = np.array([(self.Whiten ** -1) @ self.B @ self.Mixing @ yi for yi in y_new])
        q = np.array([(yi - y_hati).T @ (yi - y_hati) for (yi, y_hati) in zip(y_new, y_hat)])

        return [i2, q]

    def fit_score(self, data):
        cov = (1 / (data.shape[0] - 1) * data.T @ data)  # compute the covariance of the data
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        whiten = np.diag(eigenvalues ** -0.5) @ eigenvectors.T
        z = whiten @ data.T

        self.ica = FastICA(max_iter=100000, whiten=False, tol=0.001)
        self.Whiten = whiten
        self.X_new = self.ica.fit_transform(z.T)
        self.Mixing = self.ica.components_
        self.n_components = data.shape[1]
        self.B = (self.Mixing @ (self.Whiten ** -1)).T
        self.Cov = cov

        f_new = np.array([self.Mixing @ xi for xi in self.X_new])
        i2 = np.array([fi.T @ fi for fi in f_new])

        x_hat = np.array([(self.Whiten ** -1) @ self.B @ self.Mixing @ xi for xi in self.X_new])
        q = np.array([(xi - x_hati).T @ (xi - x_hati) for (xi, x_hati) in zip(self.X_new, x_hat)])

        return [i2, q], [self.Whiten, self.B, self.n_components, self.X_new, self.Mixing, self.ica, self.Cov]
