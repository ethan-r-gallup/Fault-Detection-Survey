from scipy import linalg
import numpy as np
import pandas as pd
from typing import List, Tuple


class CVA:
    def __init__(self, components=None):
        if isinstance(components, list):
            self.n_components = components[2]
            self.J = components[3]
            self.F = components[4]
            self.window = components[5]
        else:
            self.n_components = None
            self.J = None
            self.F = None
            self.window = None

        self.e = None
        self.z = None

    def fit(self, data: pd.DataFrame, window: int = 3) -> list:
        data = np.array(data)
        self.window = window
        mq = data.shape[1] * window
        M = data.shape[0] - (2 * window) + 1
        self.n_components = data.shape[1]

        Y_p = np.zeros((mq, M), dtype=list)
        Y_f = np.zeros((mq, M), dtype=list)

        for i in range(M):
            k = i + window
            pk = data[i:k].flatten().T
            Y_p[:, i] = pk

            fk = data[k:k+window].flatten().T
            Y_f[:, i] = fk

        sigma_pp = np.array((Y_p @ Y_p.T) * (1/(M-1)), dtype=float)
        sigma_ff = np.array((Y_f @ Y_f.T) * (1/(M-1)), dtype=float)
        sigma_fp = np.array((Y_f @ Y_p.T) * (1/(M-1)), dtype=float)

        H = (linalg.sqrtm(sigma_ff) ** (-1)) @ sigma_fp @ (linalg.sqrtm(sigma_pp) ** (-1))

        U, S, V = linalg.svd(H)

        self.J = V.T @ (linalg.sqrtm(sigma_pp) ** (-1))
        I = np.identity(V.shape[0])
        self.F = (I - V @ V.T) @ (linalg.sqrtm(sigma_pp) ** (-1))

        self.z = np.array([self.J @ pk for pk in Y_p.T])
        self.e = np.array([self.F @ pk for pk in Y_p.T])

        return [self.e, self.z, self.n_components, self.J, self.F, self.window]

    def score(self, data: pd.DataFrame) -> List[np.ndarray]:
        a = list(data.columns)
        data = np.array(data)

        mq = data.shape[1] * self.window
        M = data.shape[0] - (2 * self.window) + 1
        Y_p = np.zeros((mq, M), dtype=list)
        Y_f = np.zeros((mq, M), dtype=list)
        for i in range(M):
            k = i + self.window
            pk = data[i:k].flatten().T
            Y_p[:, i] = pk

            fk = data[k:k+self.window].flatten().T
            Y_f[:, i] = fk

        if self.J.shape[1] != Y_p.shape[0]:
            mask = np.array([])
            for i, column in enumerate(a):
                if i == 0:
                    mask = np.arange(column, column+self.window)
                else:
                    mask = np.append(mask, np.arange(column, column+self.window))

            self.J = self.J[:, mask]
            self.J = self.J[mask, :]
            self.F = self.F[:, mask]
            self.F = self.F[mask, :]

        z = np.array([self.J @ pk for pk in Y_p.T])
        e = np.array([self.F @ pk for pk in Y_p.T])

        t2 = np.array([zk.T @ zk for zk in z])
        q = np.array([ek.T @ ek for ek in e])

        return [t2, q]

    def fit_score(self, data: pd.DataFrame, window: int = 3) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        data = np.array(data)
        if not isinstance(self.window, int):
            self.window = window
        mq = data.shape[1] * window
        M = data.shape[0] - (2 * window) + 1

        Y_p = np.zeros((mq, M), dtype=list)
        Y_f = np.zeros((mq, M), dtype=list)

        for i in range(M):
            k = i + window
            pk = data[i:k].flatten().T
            Y_p[:, i] = pk

            fk = data[k:k+window].flatten().T
            Y_f[:, i] = fk

        sigma_pp = np.array((Y_p @ Y_p.T) * (1/(M-1)), dtype=float)
        sigma_ff = np.array((Y_f @ Y_f.T) * (1/(M-1)), dtype=float)
        sigma_fp = np.array((Y_f @ Y_p.T) * (1/(M-1)), dtype=float)

        H = (linalg.fractional_matrix_power(sigma_ff, -0.5)) @ sigma_fp @ (linalg.fractional_matrix_power(sigma_pp, -0.5))
        r = np.linalg.matrix_rank(H)
        U, S, V = linalg.svd(H, full_matrices=False)
        U = U[0:mq, 0:r]
        V = V[0:mq, 0:r]
        S = S[0:r-1]
        # V = V.T

        self.J = V.T @ (linalg.fractional_matrix_power(sigma_pp, -0.5))
        I = np.identity(V.shape[0])
        self.F = (I - V @ V.T) @ (linalg.fractional_matrix_power(sigma_pp, -0.5))
        self.z = np.array([self.J @ pk for pk in Y_p.T])
        self.e = np.array([self.F @ pk for pk in Y_p.T])

        self.n_components = 66
        t2 = np.array([zk.T @ zk for zk in self.z])
        q = np.array([ek.T @ ek for ek in self.e])
        return [t2, q], [self.e, self.z.T, self.n_components, self.J, self.F, self.window]
