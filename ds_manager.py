import pandas as pd
from sklearn.model_selection import train_test_split
from data_splits import DataSplits
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DSManager:
    def __init__(self, name, folds=1):
        self.name = name
        self.folds = folds
        self.init_seed = 40
        self.random_state_start = 80
        self._reset_seed()
        dataset_path = f"data/{name}.csv"
        df = pd.read_csv(dataset_path)
        df.iloc[:, -1], class_labels = pd.factorize(df.iloc[:, -1])
        scaler = MinMaxScaler()
        df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
        self.data = df.to_numpy()
        #train:validation:evaluation_train:evaluation_test = 0.45:  0.0.5:  0.50    :0.50

    def get_name(self):
        return self.name

    def count_rows(self):
        return self.data.shape[0]

    def count_features(self):
        return self.data.shape[1]-1
    
    def _shuffle(self, seed):
        self._set_seed(seed)
        shuffled_indices = np.random.permutation(self.data.shape[0])
        self._reset_seed()
        return self.data[shuffled_indices]

    def get_k_folds(self):
        for i in range(self.folds):
            seed = self.random_state_start + i
            yield self.get_all_set_X_y_from_data(seed)

    def get_all_set_X_y_from_data(self, seed):
        data = self._shuffle(seed)
        train_validation, evaluation = train_test_split(data, test_size=0.1, random_state=seed, stratify=data[:,-1])
        train, validation = train_test_split(train_validation, test_size=0.1, random_state=seed, stratify=train_validation[:,-1])
        evaluation_train, evaluation_test = train_test_split(evaluation, test_size=0.5, random_state=seed, stratify=evaluation[:,-1])
        return DataSplits(self.name, *DSManager.get_X_y_from_data(train),
                          *DSManager.get_X_y_from_data(validation),
                          *DSManager.get_X_y_from_data(evaluation_train),
                          *DSManager.get_X_y_from_data(evaluation_test)
                          )

    def _set_seed(self, seed):
        np.random.seed(seed)

    def _reset_seed(self):
        np.random.seed(self.init_seed)

    def __repr__(self):
        return self.get_name()

    @staticmethod
    def get_X_y_from_data(data):
        x = data[:, :-1]
        y = data[:, -1]
        return x, y

    @staticmethod
    def get_dataset_names():
        return [
            "indian_pines",
            "paviaU",
            "salinasA",

            "pavia",
            "salinas"
        ]


if __name__ == "__main__":
    from collections import Counter
    df = pd.read_csv("data/indian_pines.csv")
    size = len(df)

    ds = DSManager("indian_pines")
    for split in ds.get_k_folds():
        a = len(split.train_x)
        b = len(split.validation_x)
        c = len(split.evaluation_train_x)
        d = len(split.evaluation_test_x)
        tot = a + b + c + d
        print(a,b,c,d,tot,size)

        class_counts = Counter(split.train_y)
        sorted_class_counts = dict(sorted(class_counts.items()))
        for k,v in sorted_class_counts.items():
            print(f"{v:>5}", end="")
        print("")

        class_counts = Counter(split.validation_y)
        sorted_class_counts = dict(sorted(class_counts.items()))
        for k,v in sorted_class_counts.items():
            print(f"{v:>5}", end="")
        print("")

        class_counts = Counter(split.evaluation_train_y)
        sorted_class_counts = dict(sorted(class_counts.items()))
        for k,v in sorted_class_counts.items():
            print(f"{v:>5}", end="")
        print("")

        class_counts = Counter(split.evaluation_test_y)
        sorted_class_counts = dict(sorted(class_counts.items()))
        for k,v in sorted_class_counts.items():
            print(f"{v:>5}", end="")

        print("")

