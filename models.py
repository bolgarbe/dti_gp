import torch
import gpytorch
import numpy as np
from tqdm import tqdm

from data import get_batch

class ProteinEmbeddingSimilarity(torch.nn.Module):
    def __init__(self,protein_embeddings,device,gamma=0.1,learn_gamma=True):
        super(ProteinEmbeddingSimilarity,self).__init__()

        self.num_proteins = protein_embeddings.shape[0]
        gamma = torch.nn.Parameter(torch.tensor(gamma))
        self.register_parameter('gamma',gamma)
        
        self.emb = torch.tensor(protein_embeddings).to(device)
        self.Xp2 = self.emb @ self.emb.T
        self.Xpd = torch.diag(self.Xp2)
        if learn_gamma == False:
            self.gamma.requires_grad = False

    @property
    def Xp(self):
        return torch.exp(-self.gamma*(self.Xpd + torch.unsqueeze(self.Xpd,-1) - 2*self.Xp2))

class InteractionEncoder(torch.nn.Module):
    def __init__(self,num_targets,fp_dim,embedding_dim,dropout=0.2):
        super(InteractionEncoder,self).__init__()
        self.compound_encoder = torch.nn.Sequential(
            torch.nn.Linear(fp_dim,embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.protein_encoder = torch.nn.Linear(num_targets,embedding_dim)
        torch.nn.init.xavier_uniform_(self.protein_encoder.weight)

    def forward(self,Xc,Xp,row,col):
        c_rep  = self.compound_encoder(Xc)[row]
        p_rep  = self.protein_encoder(Xp)[col]
        return c_rep*p_rep

class DKLModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        vdist = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        vstr  = gpytorch.variational.VariationalStrategy(self, inducing_points, vdist, learn_inducing_locations=True)
        super(DKLModel,self).__init__(vstr)
        
        self.mean = gpytorch.means.ConstantMean()
        self.cov  = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self,x):
        m = self.mean(x)
        C = self.cov(x)
        return gpytorch.distributions.MultivariateNormal(m,C)

class DTIGP():
    def __init__(self,protein_similarity,fp_dim,num_interactions,device,embedding_dim=512,num_inducing=512):
        
        num_proteins = protein_similarity.num_proteins
        
        self.device  = device
        self.sim = protein_similarity.to(device)
        self.encoder = InteractionEncoder(num_proteins,fp_dim,embedding_dim).to(device)
        with torch.no_grad():
            inducing_points = torch.eye(num_inducing)
        self.dkl = DKLModel(inducing_points).to(device)
        self.lh  = gpytorch.likelihoods.BernoulliLikelihood().to(device)
        self.mll = gpytorch.mlls.VariationalELBO(self.lh, self.dkl, num_data=num_interactions)

    def train(self,Xc_train,y_train,num_epochs=30,batch_size=1024):
        opt = torch.optim.Adam([{'params': self.dkl.parameters(), 'lr': 5e-2},
            {'params': self.lh.parameters(), 'lr': 1e-3},
            {'params': self.encoder.parameters(), 'lr': 5e-3, 'weight_decay': 1e-4}])

        num_batches = int(np.ceil(y_train.shape[0]/batch_size))
        with gpytorch.settings.debug(False), tqdm(total=num_epochs) as pbar:
            for epoch in range(num_epochs):
                self.dkl.train()
                self.lh.train()
                self.encoder.train()

                idx = np.arange(y_train.shape[0])
                np.random.shuffle(idx)
                for b in range(num_batches):
                    batch_idx = idx[b*batch_size:(b+1)*batch_size]
                    Xc_batch, row_batch, col_batch, y_batch = get_batch(batch_idx,Xc_train,y_train)
                    
                    opt.zero_grad()
                    out = self.dkl(self.encoder(Xc_batch.to(self.device),self.sim.Xp,row_batch,col_batch))
                    loss = -self.mll(out,y_batch.to(self.device))
                    
                    loss.backward()
                    opt.step()
                
                pbar.update(1)
    
    def predict(self,Xc_test,y_test,num_samples):
        self.dkl.eval()
        self.lh.eval()
        self.encoder.eval()

        with gpytorch.settings.debug(False), gpytorch.settings.fast_pred_var(), torch.no_grad():
            batch_idx = np.arange(Xc_test.shape[0])
            Xc_batch,row_batch,col_batch,y_batch = get_batch(batch_idx,Xc_test,y_test)
            pred  = self.dkl(self.encoder(Xc_batch.to(self.device),self.sim.Xp,row_batch,col_batch))
            probs = self.lh(pred).probs.cpu().numpy()
            smps  = self.lh(pred.sample(torch.Size((num_samples,)))).probs.cpu().numpy()
            y_flt = y_batch.cpu().numpy()
        
        return probs, smps, y_flt, row_batch, col_batch