import numpy as np
import torch
import torch.nn as nn
import nodepy.linear_multistep_method as lm


# 设置随机种子
np.random.seed(0)
torch.manual_seed(0)

class Multistep_NN(nn.Module):
    def __init__(self, dt, X, layers, M, scheme):
        super(Multistep_NN, self).__init__()

        self.dt = dt
        self.X = torch.tensor(X, dtype=torch.float32) # S x N x D
        
        self.S = X.shape[0] # number of trajectories
        self.N = X.shape[1] # number of time snapshots
        self.D = X.shape[2] # number of dimensions
        
        self.M = M # number of Adams-Moulton steps
        
        self.layers = layers
        
        # Load weights
        switch = {'AM': lm.Adams_Moulton,
                  'AB': lm.Adams_Bashforth,
                  'BDF': lm.backward_difference_formula}
        method = switch[scheme](M)
        
        # 将 method.alpha 和 method.beta 转换为浮点数数组
        self.alpha = torch.tensor(-np.array(method.alpha[::-1], dtype=np.float32))
        self.beta = torch.tensor(np.array(method.beta[::-1], dtype=np.float32))
        
        # Neural Network
        self.neural_net = self.build_network()
    
    def build_network(self):
        layers = []
        num_layers = len(self.layers)
        for l in range(num_layers - 2):
            layers.append(nn.Linear(self.layers[l], self.layers[l + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(self.layers[-2], self.layers[-1]))
        return nn.Sequential(*layers)
    
    def net_F(self, X): # S x (N-M+1) x D
        X_reshaped = X.reshape(-1, self.D) # S(N-M+1) x D
        F_reshaped = self.neural_net(X_reshaped) # S(N-M+1) x D
        F = F_reshaped.reshape(self.S, -1, self.D) # S x (N-M+1) x D
        return F # S x (N-M+1) x D
    
    def net_Y(self, X): # S x N x D
        M = self.M
        Y = self.alpha[0] * X[:, M:, :] + self.dt * self.beta[0] * self.net_F(X[:, M:, :])
        for m in range(1, M + 1):
            Y = Y + self.alpha[m] * X[:, M-m:-m, :] + self.dt * self.beta[m] * self.net_F(X[:, M-m:-m, :]) # S x (N-M+1) x D
        return Y # S x (N-M+1) x D
    
    def forward(self, X): # 前向传播
        return self.net_Y(X)

    def train_model(self, n_iter, lr=0.001, betas=(0.9, 0.999), eps=1e-32,
                    print_every=1000, target_loss=1e-10, print_time=False):
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas, eps=eps)

        self.train_loss_list = []
        self.iteration_list = []
        self.times = []

        import time
        start_time = time.time()
        for epoch in range(n_iter):
            self.train()
            optimizer.zero_grad()
            y_pred = self.forward(self.X)
            loss = loss_function(y_pred, torch.zeros_like(y_pred))
            loss.backward()
            optimizer.step()

            elapsed = time.time() - start_time
            self.times.append(elapsed)
            self.train_loss_list.append(loss.detach().cpu().numpy())
            self.iteration_list.append(epoch)

            if print_every is not None and print_every > 0 and epoch % print_every == 0:
                if print_time:
                    print(f'Epoch {epoch+1}/{n_iter}, Train_loss: {loss.item(): .16e}, Time per iteration: {elapsed:.4f} seconds')
                else:
                    print(f'Epoch {epoch+1}/{n_iter}, Train_loss: {loss.item(): .16e}')

            if loss.item() < target_loss:
                print(f'Target loss reached at iteration {epoch+1}.')
                break

        return self.train_loss_list

    def predict_f(self, X_star):
        X_star_tensor = torch.tensor(X_star, dtype=torch.float32)
        with torch.no_grad():
            F_star = self.neural_net(X_star_tensor)
        return F_star.numpy()
