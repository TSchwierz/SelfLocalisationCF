'''
GPU-accelerated Recursive Least Squares (RLS) implementation using Numba CUDA.
'''

import numpy as np
from numba import cuda, float32

# -------------------------------------------------------------------
# CUDA kernels
# -------------------------------------------------------------------

@cuda.jit
def matvec(P, y, out):
    """
    out[i,0] = sum_j P[i,j] * y[j,0]
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    n = P.shape[0]
    if i < n:
        tmp = float32(0.0)
        for j in range(n):
            tmp += P[i, j] * y[j, 0]
        out[i, 0] = tmp

@cuda.jit
def at_y(A, y, out):
    """
    out[k,0] = sum_i A[i,k] * y[i,0]
    """
    k = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    nf, no = A.shape
    if k < no:
        tmp = float32(0.0)
        for i in range(nf):
            tmp += A[i, k] * y[i, 0]
        out[k, 0] = tmp

@cuda.jit
def outer(u, v, M):
    """
    M[i,j] += u[i,0] * v[0,j]
    """
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    nf, nj = M.shape
    if i < nf and j < nj:
        M[i, j] += u[i, 0] * v[0, j]

@cuda.jit
def update_P(P, KyTP, lam):
    """
    P = (P - KyTP) / lam
    """
    i, j = cuda.grid(2)
    nf, _ = P.shape
    if i < nf and j < nf:
        P[i, j] = (P[i, j] - KyTP[i, j]) / lam

# -------------------------------------------------------------------
# Host-side class with GPU-accelerated update
# -------------------------------------------------------------------

class OptimisedRLS_GPU:
    def __init__(self, num_features, num_outputs,
                 lam=0.99, delta=1.0, eps=1e-10):
        self.nf = num_features
        self.no = num_outputs
        self.lam = lam
        self.eps = eps

        # Host versions (for occasional read-back / init)
        A = np.zeros((num_features, num_outputs), dtype=np.float32)
        P = np.eye(num_features, dtype=np.float32) * delta

        # Copy to device
        self.d_A = cuda.to_device(A)
        self.d_P = cuda.to_device(P)

        # Pre-allocate device temporaries
        self.d_Py = cuda.device_array((num_features, 1), dtype=np.float32)
        self.d_K = cuda.device_array((num_features, 1), dtype=np.float32)
        self.d_error = cuda.device_array((num_outputs, 1), dtype=np.float32)
        self.d_KyTP = cuda.device_array((num_features, num_features), dtype=np.float32)

        # Configure kernel launch parameters
        threads = 128
        self.blocks_1d = ((num_features + threads - 1) // threads,)
        self.blocks_out = ((num_features + threads - 1) // threads,
                           (num_outputs + threads - 1) // threads)
        self.blocks_2d = ((num_features + 15) // 16, (num_features + 15) // 16)
        self.threads_2d = (16, 16)

    def update(self, y_host, x_host):
        # Reshape & copy new data
        y = y_host.astype(np.float32).reshape(self.nf, 1)
        x = x_host.astype(np.float32).reshape(self.no, 1)
        d_y = cuda.to_device(y)
        d_x = cuda.to_device(x)

        # 1) Py = P @ y
        matvec[self.blocks_1d, 128](self.d_P, d_y, self.d_Py)

        # 2) yTPy = y^T @ Py (+ eps), denom = lam + yTPy
        Py = self.d_Py.copy_to_host()
        yTPy = float(np.dot(y.T, Py)) + self.eps
        denom = self.lam + yTPy

        # 3) K = Py / denom
        self.d_K.copy_to_device(Py / denom)

        # 4) error = x - A.T @ y
        at_y[self.blocks_1d, 128](self.d_A, d_y, self.d_error)
        err = d_x.copy_to_host() - self.d_error.copy_to_host()
        self.d_error.copy_to_device(err)

        # 5) A += K @ error^T
        outer[self.blocks_out, (16, 16)](self.d_K, self.d_error, self.d_A)

        # 6) KyTP = K @ (y^T @ P)
        Py_T = Py.T.reshape(1, self.nf)
        d_PyT = cuda.to_device(Py_T)
        outer[self.blocks_2d, self.threads_2d](self.d_K, d_PyT, self.d_KyTP)

        # 7) P = (P - KyTP) / lam
        update_P[self.blocks_2d, self.threads_2d](self.d_P, self.d_KyTP, self.lam)

        # Return updated A back to host
        return self.d_A.copy_to_host()

    def predict(self, y_host):
        y = y_host.astype(np.float32).reshape(self.nf, 1)
        d_y = cuda.to_device(y)
        d_out = cuda.device_array((self.no, 1), dtype=np.float32)
        at_y[self.blocks_1d, 128](self.d_A, d_y, d_out)
        return d_out.copy_to_host().flatten()
