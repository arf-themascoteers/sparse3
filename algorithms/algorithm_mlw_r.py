from algorithm import Algorithm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math
from data_splits import DataSplits
import train_test_evaluator


class Sparse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return torch.relu(X)


class ZhangNet(nn.Module):
    def __init__(self, bands, number_of_classes, last_layer_input, device):
        super().__init__()
        torch.manual_seed(3)
        self.device = device
        self.bands = bands
        self.number_of_classes = number_of_classes
        self.last_layer_input = last_layer_input
        self.heads = 8
        cw = torch.randn(self.heads, self.bands, dtype=torch.float32).unsqueeze(1).to(self.device)
        self.channel_weights = nn.Parameter(cw)
        self.classnets = nn.ModuleList([nn.Sequential(
            nn.Linear(self.bands, 200),
            nn.LeakyReLU(),
            nn.Linear(200, self.number_of_classes),
        ) for i in range(self.heads+1)])
        self.sparse = Sparse()
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    def forward(self, iX):
        X = iX.unsqueeze(0)
        mean_channel_weights = torch.mean(self.channel_weights, dim=0, keepdim=True)
        all_channel_weights = torch.cat((self.channel_weights, mean_channel_weights), dim=0)
        sparse_channel_weights = self.sparse(all_channel_weights)
        recalibrated_X = X*sparse_channel_weights

        modules = list(self.classnets)
        inputs = [recalibrated_X[i] for i in range(recalibrated_X.shape[0])]

        #output = [self.classnets[i](inputs[i]) for i in range(self.heads+1)]
        output = torch.nn.parallel.parallel_apply(modules, inputs)
        output = torch.stack(output, dim=0)

        return mean_channel_weights.squeeze(0).squeeze(0), sparse_channel_weights[-1].squeeze(0), output


class Algorithm_mlw_r(Algorithm):
    def __init__(self, target_size:int, splits:DataSplits, tag, reporter, verbose, fold):
        super().__init__(target_size, splits, tag, reporter, verbose, fold)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.class_size = len(np.unique(self.splits.train_y))
        self.last_layer_input = 100
        self.zhangnet = ZhangNet(self.splits.train_x.shape[1], self.class_size, self.last_layer_input, self.device).to(self.device)
        self.total_epoch = 2000
        self.epoch = -1
        self.X_train = torch.tensor(self.splits.train_x, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(self.splits.train_y, dtype=torch.int32).to(self.device)
        self.X_val = torch.tensor(self.splits.validation_x, dtype=torch.float32).to(self.device)
        self.y_val = torch.tensor(self.splits.validation_y, dtype=torch.int32).to(self.device)

    def get_selected_indices(self):
        optimizer = torch.optim.Adam(self.zhangnet.parameters(), lr=0.001, betas=(0.9,0.999))
        dataset = TensorDataset(self.X_train, self.y_train)
        dataloader = DataLoader(dataset, batch_size=128000, shuffle=True)
        channel_weights = None
        loss = 0
        l1_loss = 0
        mse_loss = 0

        for epoch in range(self.total_epoch):
            self.epoch = epoch
            for batch_idx, (X, y) in enumerate(dataloader):
                #print(epoch, self.zhangnet.channel_weights[:,0,0].tolist())
                optimizer.zero_grad()
                channel_weights, sparse_weights, y_hat = self.zhangnet(X)
                mean_weight, all_bands, selected_bands = self.get_indices(sparse_weights)
                self.set_all_indices(all_bands)
                self.set_selected_indices(selected_bands)
                self.set_weights(mean_weight)

                y = y.type(torch.LongTensor).to(self.device)
                mse_loss = 0
                for i in range(y_hat.shape[0]):
                    weight = 1
                    if i == self.zhangnet.heads - 1:
                        weight = self.zhangnet.heads
                    mse_loss = mse_loss + (weight * self.criterion(y_hat[i], y))

                mse_loss = mse_loss / (self.zhangnet.heads*2)
                lambda1 = self.get_lambda1(epoch)
                lambda2 = self.get_lambda2(epoch)
                l1_loss = self.l1(channel_weights)
                l2_loss = self.l2(channel_weights)
                loss = mse_loss + lambda1*l1_loss + lambda2/l2_loss
                if batch_idx == 0 and self.epoch%10 == 0:
                    self.report_stats(channel_weights, sparse_weights, epoch, mse_loss,
                                      l1_loss.item(), lambda1,
                                      l2_loss.item(), lambda2,
                                      loss)
                loss.backward()
                optimizer.step()

        print("Zhang - selected bands and weights:")
        print("".join([str(i).ljust(10) for i in self.selected_indices]))
        return self.zhangnet, self.selected_indices

    def report_stats(self, channel_weights, sparse_weights, epoch, mse_loss, l1_loss, lambda1, l2_loss, lambda2, loss):
        _, _,y_hat = self.zhangnet(self.X_train)
        y_hat = y_hat[-1]
        yp = torch.argmax(y_hat, dim=1)
        yt = self.y_train.cpu().detach().numpy()
        yh = yp.cpu().detach().numpy()
        t_oa, t_aa, t_k = train_test_evaluator.calculate_metrics(yt, yh)

        _, _,y_hat = self.zhangnet(self.X_val)
        y_hat = y_hat[-1]
        yp = torch.argmax(y_hat, dim=1)
        yt = self.y_val.cpu().detach().numpy()
        yh = yp.cpu().detach().numpy()
        v_oa, v_aa, v_k = train_test_evaluator.calculate_metrics(yt, yh)

        mean_weight = channel_weights
        means_sparse = sparse_weights
        min_cw = torch.min(mean_weight).item()
        min_s = torch.min(means_sparse).item()
        max_cw = torch.max(mean_weight).item()
        max_s = torch.max(means_sparse).item()
        avg_cw = torch.mean(torch.abs(mean_weight)).item()
        avg_s = torch.mean(torch.abs(means_sparse)).item()

        l0_cw = torch.norm(mean_weight, p=0).item()
        l0_s = torch.norm(means_sparse, p=0).item()

        mean_weight, all_bands, selected_bands = self.get_indices(channel_weights)

        oa, aa, k = train_test_evaluator.evaluate_split(self.splits, self)
        self.reporter.report_epoch(epoch, mse_loss, l1_loss, lambda1, l2_loss,lambda2,loss,
                                   t_oa, t_aa, t_k,
                                   v_oa, v_aa, v_k,
                                   oa, aa, k,
                                   min_cw, max_cw, avg_cw,
                                   min_s, max_s, avg_s,
                                   l0_cw, l0_s,
                                   selected_bands, means_sparse)

    def get_indices(self, sparse_weights):
        band_indx = (torch.argsort(torch.abs(sparse_weights), descending=True)).tolist()
        return sparse_weights, band_indx, band_indx[: self.target_size]

    def l1(self, channel_weights):
        return torch.norm(channel_weights, p=1)

    def l2(self, channel_weights):
        return torch.norm(channel_weights, p=2)

    def get_lambda1(self, epoch):
        return 0.001

    def get_lambda2(self, epoch):
        return 0





