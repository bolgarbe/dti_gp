import torch
import numpy as np

def get_batch(batch_idx,X,y):
    X_batch = X[batch_idx].tocoo()
    X_batch_tensor = torch.sparse_coo_tensor(torch.tensor(np.vstack([X_batch.row,X_batch.col])),torch.tensor(X_batch.data),size=X_batch.shape)
    y_batch = y[batch_idx].tocoo()
    return X_batch_tensor, y_batch.row, y_batch.col, torch.tensor((y_batch.data.astype(np.float32)+1)/2)