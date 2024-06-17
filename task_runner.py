import torch

from ds_manager import DSManager
from reporter import Reporter
import pandas as pd
from metrics import Metrics
from algorithm import Algorithm
import train_test_evaluator


class TaskRunner:
    def __init__(self, task, folds=1, tag="results", skip_all_bands=False, verbose=False):
        self.task = task
        self.folds = folds
        self.skip_all_bands = skip_all_bands
        self.verbose = verbose
        self.tag = tag
        self.reporter = Reporter(self.tag, self.skip_all_bands)
        self.cache = pd.DataFrame(columns=["dataset","fold","algorithm",
                                           "oa","aa","k","time","selected_features","selected_weights"])

    def evaluate(self):
        for dataset_name in self.task["datasets"]:
            dataset = DSManager(name=dataset_name, folds=self.folds)
            if not self.skip_all_bands:
                self.evaluate_for_all_features(dataset)
            for fold, splits in enumerate(dataset.get_k_folds()):
                for algorithm in self.task["algorithms"]:
                    for target_size in self.task["target_sizes"]:
                        print(algorithm)
                        algorithm_object = Algorithm.create(algorithm, target_size, splits, self.tag, self.reporter, self.verbose, fold)
                        self.process_a_case(algorithm_object, fold)

        return self.reporter.get_summary(), self.reporter.get_details()

    def process_a_case(self, algorithm:Algorithm, fold):
        metric = self.reporter.get_saved_metrics(algorithm)
        if metric is None:
            metric = self.get_results_for_a_case(algorithm, fold)
            self.reporter.write_details(algorithm, metric)
            if algorithm.weights is not None:
                weights = torch.abs(algorithm.weights)
                for i,w in enumerate(weights):
                    print(i+1, round(w.item(),4))


    def get_results_for_a_case(self, algorithm:Algorithm, fold):
        metric = self.get_from_cache(algorithm, fold)
        if metric is not None:
            print(f"Selected features got from cache for {algorithm.splits.get_name()} for size {algorithm.target_size} for fold {fold} for {algorithm.get_name()}")
            algorithm.set_selected_indices(metric.selected_features)
            return algorithm.compute_performance()
        print(f"Computing {algorithm.get_name()} {algorithm.splits.get_name()} Fold {fold}")
        metric = algorithm.compute_performance()
        self.save_to_cache(algorithm, fold, metric)
        return metric

    def save_to_cache(self, algorithm, fold, metric:Metrics):
        if not algorithm.is_cacheable():
            return
        self.cache.loc[len(self.cache)] = {
            "dataset":algorithm.splits.get_name(),
            "algorithm": algorithm.get_name(),
            "fold": fold,
            "time":metric.time,"oa":metric.oa,"aa":metric.aa,"k":metric.k, "selected_features":algorithm.get_all_indices()
        }

    def get_from_cache(self, algorithm:Algorithm, fold):
        if not algorithm.is_cacheable():
            return None
        if len(self.cache) == 0:
            return None
        rows = self.cache.loc[
            (self.cache["dataset"] == algorithm.splits.get_name()) &
            (self.cache["fold"] == fold) &
            (self.cache["algorithm"] == algorithm.get_name())
        ]
        if len(rows) == 0:
            return None
        row = rows.iloc[0]
        selected_features = row["selected_features"][0:algorithm.target_size]
        return Metrics(row["time"], row["oa"],row["aa"], row["k"], selected_features)

    def evaluate_for_all_features(self, dataset):
        for fold, splits in enumerate(dataset.get_k_folds()):
            self.evaluate_for_all_features_fold(fold, splits)

    def evaluate_for_all_features_fold(self, fold, splits):
        oa, aa, k = self.reporter.get_saved_metrics_for_all_feature(splits.get_name())
        if oa is not None and k is not None:
            print(f"Fold {fold} for {splits.get_name()} was done")
            return
        oa, aa, k = train_test_evaluator.evaluate_split(splits)
        self.reporter.write_details_all_features(fold, splits.get_name(), oa, aa, k)
