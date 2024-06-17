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
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.k = 0

    def forward(self, X):
        X = torch.where(X < self.k, 0, X)
        return X


class ZhangNet(nn.Module):
    def __init__(self, bands, number_of_classes):
        super().__init__()
        torch.manual_seed(3)
        self.bands = bands
        self.number_of_classes = number_of_classes
        self.weighter = nn.Parameter(torch.ones(self.bands)/2, requires_grad=True)
        self.classnet = nn.Sequential(
            nn.Linear(self.bands, 100),
            nn.LeakyReLU(),
            nn.Linear(100,self.number_of_classes)
        )
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)

    def forward(self, X):
        reweight_out = self.weighter * X
        output = self.classnet(reweight_out)
        return self.weighter, self.weighter, output


class Algorithm_lw(Algorithm):
    def __init__(self, target_size:int, splits:DataSplits, tag, reporter, verbose, fold):
        super().__init__(target_size, splits, tag, reporter, verbose, fold)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.class_size = len(np.unique(self.splits.train_y))
        self.zhangnet = ZhangNet(self.splits.train_x.shape[1], self.class_size).to(self.device)
        self.total_epoch = 1600
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
                optimizer.zero_grad()
                channel_weights, sparse_weights, y_hat = self.zhangnet(X)

                mean_weight, all_bands, selected_bands = self.get_indices(channel_weights)
                self.set_all_indices(all_bands)
                self.set_selected_indices(selected_bands)

                y = y.type(torch.LongTensor).to(self.device)
                mse_loss = self.criterion(y_hat, y)
                l1_loss = self.l1_loss(channel_weights)
                l2_loss = torch.mean(channel_weights*channel_weights)
                alpha = 0
                #lambda_value = self.get_lambda(epoch+1)
                lambda_value = 1
                loss = mse_loss + lambda_value*l1_loss + alpha* (-l2_loss)
                if batch_idx == 0 and self.epoch%10 == 0:
                    self.report_stats(channel_weights, sparse_weights, epoch, mse_loss, l1_loss, lambda_value, l2_loss, alpha, loss)
                loss.backward()
                optimizer.step()

        print("Zhang - selected bands and weights:")
        print("".join([str(i).ljust(10) for i in self.selected_indices]))

        return self.zhangnet, self.selected_indices

    def report_stats(self, channel_weights, sparse_weights, epoch, mse_loss, l1_loss, lambda_value, l2_loss, alpha, loss):
        _, _, y_hat = self.zhangnet(self.X_train)
        yp = torch.argmax(y_hat, dim=1)
        yt = self.y_train.cpu().detach().numpy()
        yh = yp.cpu().detach().numpy()
        t_oa, t_aa, t_k = train_test_evaluator.calculate_metrics(yt, yh)

        _, _, y_hat = self.zhangnet(self.X_val)
        yp = torch.argmax(y_hat, dim=1)
        yt = self.y_val.cpu().detach().numpy()
        yh = yp.cpu().detach().numpy()
        v_oa, v_aa, v_k = train_test_evaluator.calculate_metrics(yt, yh)

        mean_weight = channel_weights
        mean_sparse = sparse_weights

        min_cw = torch.min(mean_weight).item()
        min_s = torch.min(mean_sparse).item()
        max_cw = torch.max(mean_weight).item()
        max_s = torch.max(mean_sparse).item()
        avg_cw = torch.mean(mean_weight).item()
        avg_s = torch.mean(mean_sparse).item()

        l0_cw = torch.norm(mean_weight, p=0).item()
        l0_s = torch.norm(mean_sparse, p=0).item()

        mean_weight, all_bands, selected_bands = self.get_indices(channel_weights)

        oa, aa, k = train_test_evaluator.evaluate_split(self.splits, self)
        means_sparse = sparse_weights

        self.reporter.report_epoch(epoch, mse_loss, l1_loss, lambda_value, l2_loss, alpha, loss,
                                   t_oa, t_aa, t_k,
                                   v_oa, v_aa, v_k,
                                   oa, aa, k,
                                   min_cw, max_cw, avg_cw,
                                   min_s, max_s, avg_s,
                                   l0_cw, l0_s,
                                   selected_bands, means_sparse)

    def get_indices(self, channel_weights):
        channel_weights = torch.abs(channel_weights)
        band_indx = (torch.argsort(channel_weights, descending=True)).tolist()
        return channel_weights, band_indx, band_indx[: self.target_size]

    def l1_loss(self, channel_weights):
        return torch.mean(torch.abs(channel_weights))

    def get_lambda(self, epoch):
        if epoch < 200:
            return 0.01
        else:
            return 0.1
            #return 0.001 * (epoch - start) / self.total_epoch




