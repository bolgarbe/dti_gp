import numpy as np
from scipy.sparse import load_npz
import torch

from models import DTIGP, ProteinEmbeddingSimilarity

dev = torch.device('cuda:0')

y  = load_npz("data/kiba_all_BINARY.npz")
Xc = load_npz("data/kiba_ecfp_32000_FINGERPRINT.npz").astype(np.float32)
Xp = np.load("data/kiba_cpc_VECTOR.npy").astype(np.float32)
f  = np.load("data/kiba_compound_folding_SPLIT.npy")

train_idx = np.isin(f,[1,2,3,4])
test_idx  = f == 0
Xc_train  = Xc[train_idx]
Xc_test   = Xc[test_idx]
y_train   = y[train_idx]
y_test    = y[test_idx]

sim = ProteinEmbeddingSimilarity(Xp,dev,gamma=10.0,learn_gamma=False)
model = DTIGP(sim,Xc_train.shape[1],y_train.nnz,dev)

model.train(Xc_train,y_train,num_epochs=5)
probs, samples, y_ref, rows, cols = model.predict(Xc_test,y_test,1000)

from sklearn.metrics import roc_curve, precision_recall_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
from matplotlib import pyplot as plt

fpr,tpr,_ = roc_curve(y_ref,probs)
prc,rec,_ = precision_recall_curve(y_ref,probs)
print("AUROC={:.3f}, AUPR={:.3f}".format(auc(fpr,tpr),auc(rec,prc)))

prob_true, prob_pred = calibration_curve(y_ref,probs,n_bins=20)
plt.scatter(prob_pred,prob_true)
plt.plot([0,1],[0,1],ls='dotted',c='black')
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('Predicted probability')
plt.ylabel('Fraction of positives')
plt.savefig('calibration.jpg',bbox_inches='tight')
