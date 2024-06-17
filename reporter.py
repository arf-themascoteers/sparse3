import os
import pandas as pd
from metrics import Metrics


class Reporter:
    def __init__(self, tag="results", skip_all_bands=False):
        self.tag = tag
        self.skip_all_bands = skip_all_bands
        self.summary_filename = f"summary_{tag}.csv"
        self.details_filename = f"details_{tag}.csv"
        self.summary_file = os.path.join("results", self.summary_filename)
        self.details_file = os.path.join("results", self.details_filename)

        self.current_fold = -1

        self.current_epoch_report_file = None

        os.makedirs("results", exist_ok=True)

        if not os.path.exists(self.summary_file):
            with open(self.summary_file, 'w') as file:
                file.write("dataset,target_size,algorithm,time,oa,aa,k,selected_features\n")

        if not os.path.exists(self.details_file):
            with open(self.details_file, 'w') as file:
                file.write("dataset,target_size,algorithm,time,oa,aa,k,selected_features,fold\n")

        if not self.skip_all_bands:
            self.all_features_details_filename = f"all_features_details_{self.summary_filename}"
            self.all_features_summary_filename = f"all_features_summary_{self.summary_filename}"
            self.all_features_summary_file = os.path.join("results", self.all_features_summary_filename)
            self.all_features_details_file = os.path.join("results", self.all_features_details_filename)

            if not os.path.exists(self.all_features_summary_file):
                with open(self.all_features_summary_file, 'w') as file:
                    file.write("dataset,oa,aa,k\n")

            if not os.path.exists(self.all_features_details_file):
                with open(self.all_features_details_file, 'w') as file:
                    file.write("fold,dataset,oa,aa,k\n")

    def get_summary(self):
        return self.summary_file

    def get_details(self):
        return self.details_file

    def write_details(self, algorithm, metric:Metrics):
        time = Reporter.sanitize_metric(metric.time)
        oa = Reporter.sanitize_metric(metric.oa)
        aa = Reporter.sanitize_metric(metric.aa)
        k = Reporter.sanitize_metric(metric.k)
        metric.selected_features = sorted(metric.selected_features)
        with open(self.details_file, 'a') as file:
            file.write(f"{algorithm.splits.get_name()},{algorithm.target_size},{algorithm.get_name()},"
                       f"{time},{oa},{aa},{k},{'|'.join([str(i) for i in metric.selected_features])},{self.current_fold}\n")
        self.update_summary(algorithm)

    def update_summary(self, algorithm):
        df = pd.read_csv(self.details_file)
        df = df[(df["dataset"] == algorithm.splits.get_name()) & (df["algorithm"] == algorithm.get_name()) & (df["target_size"] == algorithm.target_size)]
        if len(df) == 0:
            return
        time = round(df["time"].mean(), 2)
        oa = Reporter.sanitize_metric(df["oa"].mean())
        aa = Reporter.sanitize_metric(df["aa"].mean())
        k = Reporter.sanitize_metric(df["k"].mean())
        selected_features = '||'.join(df['selected_features'].astype(str))

        df2 = pd.read_csv(self.summary_file)
        mask = ((df2["dataset"] == algorithm.splits.get_name()) & (df2["target_size"] == algorithm.target_size) & (df2["algorithm"] == algorithm.get_name()))
        if len(df2[mask]) == 0:
            df2.loc[len(df2)] = {
                "dataset":algorithm.splits.get_name(), "target_size":algorithm.target_size, "algorithm": algorithm.get_name(),
                "time":time,"oa":oa,"aa":aa,"k":k, "selected_features":selected_features
            }
        else:
            df2.loc[mask, 'time'] = time
            df2.loc[mask, 'oa'] = oa
            df2.loc[mask, 'aa'] = aa
            df2.loc[mask, 'k'] = k
            df2.loc[mask, 'selected_features'] = selected_features
        df2.to_csv(self.summary_file, index=False)

    def write_details_all_features(self, fold, dataset, oa, aa, k):
        oa = Reporter.sanitize_metric(oa)
        aa = Reporter.sanitize_metric(aa)
        k = Reporter.sanitize_metric(k)
        with open(self.all_features_details_file, 'a') as file:
            file.write(f"{fold},{dataset},{oa},{aa},{k}\n")
        self.update_summary_for_all_features(dataset)

    def update_summary_for_all_features(self, dataset):
        df = pd.read_csv(self.all_features_details_file)
        df = df[df["dataset"] == dataset]
        if len(df) == 0:
            return

        oa = max(df["oa"].mean(),0)
        aa = max(df["aa"].mean(),0)
        k = max(df["k"].mean(),0)

        df2 = pd.read_csv(self.all_features_summary_file)
        mask = (df2['dataset'] == dataset)
        if len(df2[mask]) == 0:
            df2.loc[len(df2)] = {"dataset":dataset, "oa":oa, "aa":aa, "k": k}
        else:
            df2.loc[mask, 'oa'] = oa
            df2.loc[mask, 'aa'] = aa
            df2.loc[mask, 'k'] = k
        df2.to_csv(self.all_features_summary_file, index=False)

    def get_saved_metrics(self, algorithm):
        df = pd.read_csv(self.details_file)
        if len(df) == 0:
            return None
        rows = df.loc[(df["dataset"] == algorithm.splits.get_name()) & (df["target_size"] == algorithm.target_size) &
                      (df["fold"] == self.current_fold) & (df["algorithm"] == algorithm.get_name())
                      ]
        if len(rows) == 0:
            return None
        row = rows.iloc[0]
        return Metrics(row["time"], row["oa"], row["aa"], row["k"], row["selected_features"])

    def get_saved_metrics_for_all_feature(self, dataset):
        df = pd.read_csv(self.all_features_details_file)
        if len(df) == 0:
            return None, None, None
        rows = df.loc[(df['fold'] == self.current_fold) & (df['dataset'] == dataset)]
        if len(rows) == 0:
            return None, None, None
        row = rows.iloc[0]
        return row["oa"], row["aa"], row["k"]

    @staticmethod
    def sanitize_metric(metric):
        return round(max(metric, 0),3)

    @staticmethod
    def sanitize_weight(metric):
        return round(max(metric, 0),5)

    def create_epoch_report(self, tag, algorithm, dataset, target_size, fold):
        self.current_fold = fold
        self.current_epoch_report_file = os.path.join("results", f"{tag}_{algorithm}_{dataset}_{target_size}_{self.current_fold}.csv")

    def report_epoch(self, epoch, mse_loss, l1_loss, lambda_value, l2_loss, alpha, loss,
                     t_oa,t_aa,t_k,
                     v_oa,v_aa,v_k,
                     oa,aa,k,
                     min_cw, max_cw, avg_cw,
                     min_s, max_s, avg_s,
                     l0_cw, l0_s,
                     selected_bands, mean_weight):
        if not os.path.exists(self.current_epoch_report_file):
            with open(self.current_epoch_report_file, 'w') as file:
                weight_labels = list(range(len(mean_weight)))
                weight_labels = [f"weight_{i}" for i in weight_labels]
                weight_labels = ",".join(weight_labels)
                file.write(f"epoch,mse_loss,l1_loss,lambda_value,l2_loss,alpha,loss,"
                           f"t_oa,t_aa,t_k,"
                           f"v_oa,v_aa,v_k,"
                           f"oa,aa,k,"
                           f"min_cw,max_cw,avg_cw,"
                           f"min_s,max_s,avg_s,"
                           f"l0_cw,l0_s,"
                           f"selected_bands,selected_weights,{weight_labels}\n")
        with open(self.current_epoch_report_file, 'a') as file:
            weights = [str(Reporter.sanitize_weight(i.item())) for i in mean_weight]
            weights = ",".join(weights)
            selected_bands_str = '|'.join([str(i) for i in selected_bands])

            selected_weights = [str(Reporter.sanitize_weight(i.item())) for i in mean_weight[selected_bands]]
            selected_weights_str = '|'.join(selected_weights)

            file.write(f"{epoch},{Reporter.sanitize_metric(mse_loss)},{l1_loss},{lambda_value},{l2_loss},{alpha},{Reporter.sanitize_metric(loss)},"
                       f"{Reporter.sanitize_metric(t_oa)},{Reporter.sanitize_metric(t_aa)},{Reporter.sanitize_metric(t_k)},"
                       f"{Reporter.sanitize_metric(v_oa)},{Reporter.sanitize_metric(v_aa)},{Reporter.sanitize_metric(v_k)},"                    
                       f"{Reporter.sanitize_metric(oa)},{Reporter.sanitize_metric(aa)},{Reporter.sanitize_metric(k)},"
                       f"{Reporter.sanitize_weight(min_cw)},{Reporter.sanitize_weight(max_cw)},{Reporter.sanitize_weight(avg_cw)},"
                       f"{Reporter.sanitize_weight(min_s)},{Reporter.sanitize_weight(max_s)},{Reporter.sanitize_weight(avg_s)},"
                       f"{int(l0_cw)},{int(l0_s)},"
                       f"{selected_bands_str},{selected_weights_str},{weights}\n")

