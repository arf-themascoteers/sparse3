class DataSplits:
    def __init__(self,
                 name,
                 train_x, train_y,
                 validation_x, validation_y,
                 evaluation_train_x, evaluation_train_y,
                 evaluation_test_x, evaluation_test_y):
        self.name = name
        self.train_x = train_x
        self.train_y = train_y
        self.validation_x = validation_x
        self.validation_y = validation_y
        self.evaluation_train_x = evaluation_train_x
        self.evaluation_train_y = evaluation_train_y
        self.evaluation_test_x = evaluation_test_x
        self.evaluation_test_y = evaluation_test_y

    def get_name(self):
        return self.name

    def splits_description(self, short=True):
        desc = f"train={len(self.train_y)}; valid={len(self.validation_y)}; " \
               f"evaluation_train={len(self.evaluation_train_y)}; evaluation_test={len(self.evaluation_test_y)};\n"
        if not short:
            desc = f"{desc}train_x={self.train_x[0:3,0]}; train_y={self.train_y[0:3]};\n"
            desc = f"{desc}validation_x={self.validation_x[0:3,0]}; validation_y={self.validation_y[0:3]};\n"
            desc = f"{desc}evaluation_train_x={self.evaluation_train_x[0:3,0]}; evaluation_train_y={self.evaluation_train_y[0:3]};\n"
            desc = f"{desc}evaluation_test_x={self.evaluation_test_x[0:3,0]}; evaluation_test_y={self.evaluation_test_y[0:3]};\n"
        return desc