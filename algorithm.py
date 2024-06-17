from abc import ABC, abstractmethod
from data_splits import DataSplits
from metrics import Metrics
from datetime import datetime
import train_test_evaluator
import torch
import importlib


class Algorithm(ABC):
    def __init__(self, target_size:int, splits:DataSplits, tag, reporter, verbose, fold):
        self.target_size = target_size
        self.splits = splits
        self.tag = tag
        self.reporter = reporter
        self.verbose = verbose
        self.selected_indices = None
        self.model = None
        self.all_indices = None
        self.reporter.create_epoch_report(tag, self.get_name(), self.splits.get_name(), self.target_size, fold)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self):
        self.model, self.selected_indices = self.get_selected_indices()
        return self.selected_indices

    def transform(self, X):
        if len(self.selected_indices) != 0:
            return self.transform_with_selected_indices(X)
        return self.model.transform(X)

    def transform_with_selected_indices(self, X):
        return X[:,self.selected_indices]

    def compute_performance(self):
        start_time = datetime.now()
        if self.selected_indices is None:
            self.fit()
        elapsed_time = (datetime.now() - start_time).total_seconds()
        oa, aa, k = train_test_evaluator.evaluate_split(self.splits, self)
        return Metrics(elapsed_time, oa, aa, k, self.selected_indices)

    @abstractmethod
    def get_selected_indices(self):
        pass

    def get_name(self):
        class_name = self.__class__.__name__
        name_part = class_name[len("Algorithm_"):]
        return name_part

    def get_all_indices(self):
        return self.all_indices

    def set_all_indices(self, all_indices):
        self.all_indices = all_indices

    def set_selected_indices(self, selected_indices):
        self.selected_indices = selected_indices

    def is_cacheable(self):
        return True

    @staticmethod
    def create(name, target_size, splits, tag, reporter, verbose, fold):
        class_name = f"Algorithm_{name}"
        module = importlib.import_module(f"algorithms.algorithm_{name}")
        clazz = getattr(module, class_name)
        return clazz(target_size, splits, tag, reporter, verbose, fold)
