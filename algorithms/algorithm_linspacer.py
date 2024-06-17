from algorithm import Algorithm
import linspacer
import torch
from data_splits import DataSplits


class Algorithm_linspacer(Algorithm):
    def __init__(self, target_size:int, splits:DataSplits, tag, reporter, verbose, fold):
        super().__init__(target_size, splits, tag, reporter, verbose, fold)
        self.indices = None

    def get_selected_indices(self):
        original_size = self.splits.train_x.shape[1]
        indices = linspacer.get_points(0, original_size-1, self.target_size,1)
        self.indices = torch.round(indices).long().tolist()
        return self, self.indices

    def transform(self, X):
        return X[:,self.indices]

    def is_cacheable(self):
        return False
