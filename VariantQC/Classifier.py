import os
import sys
import time

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from VariantQC.loadvcf import VCFExtract, VCF2CSV
from VariantQC.utils import Metrics


class VCFDataset:
    def __init__(self, csv_filepath):
        self.vcf = pd.read_csv(csv_filepath, converters={'chrom': str})
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_valid = None
        self.y_test = None
        self.y_valid = None

    def deal_missing_data(self):
        print("Deal with missing data")
        print("Data missing info: ")
        print(self.vcf.isna().sum())  # 输出缺失数据情况

        data_count = self.vcf.shape[0]
        for feature in self.vcf.columns:
            if self.vcf[feature].isna().sum() / data_count > 0.2 or feature == "ExcessHet":
                self.vcf = self.vcf.drop(labels=feature, axis=1)

    def impute_data(self, df, strategy='median'):
        print("Impute missing data")
        df_median = SimpleImputer(missing_values=np.nan, strategy=strategy)
        df = df_median.fit_transform(df)
        return pd.DataFrame(df)

    def split_x_y(self):
        self.x = self.vcf.iloc[:, 2:-1]
        self.y = self.vcf.iloc[:, -1]

    def standard(self, df):
        from sklearn import preprocessing
        return preprocessing.StandardScaler().fit_transform(df)

    def get_dataset(self):
        VCFDataset.deal_missing_data(self)
        VCFDataset.split_x_y(self)

        x_train_val, self.x_test, y_train_val, self.y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                              random_state=0)
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(x_train_val, y_train_val,
                                                                                  test_size=0.25, random_state=0)

        self.x_train = VCFDataset.standard(self, VCFDataset.impute_data(self, self.x_train))
        self.x_test = VCFDataset.standard(self, VCFDataset.impute_data(self, self.x_test))


class Classifier:
    def __init__(self, n_trees=150, kind="GB"):
        if kind.upper() == "RF" or kind.upper() == "RANDOMFOREST":
            self.kind = "RF"
            self.clf = RandomForestClassifier(criterion='gini', max_depth=5, n_estimators=n_trees)
        elif kind.upper() == "AB" or kind.upper() == "ADABOOST":
            self.kind = "AB"
            self.clf = AdaBoostClassifier(n_estimators=n_trees)
        elif kind.upper() == "GB" or kind.upper() == "GRADIENTBOOST":
            self.kind = "GB"
            self.clf = GradientBoostingClassifier(n_estimators=n_trees)
        elif kind.upper() == "LR" or kind.upper() == "LogisticRegression":
            self.kind = "LR"
            self.clf = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced')
        else:
            raise Exception("No such type of classifier exist.")

    def fit(self, X, y, sample_weight=None):
        print("Begin training model")
        t0 = time.time()
        self.clf.fit(X, y, sample_weight=sample_weight)
        print("Training a {}".format(self.kind))
        if (self.kind != "LR"):
            print("Importance: {}".format(self.clf.feature_importances_))
        t1 = time.time()
        print("Finish training model")
        print("Elapsed time {:.3f}s".format(t1 - t0))

    def gridsearch(self, X, y, k_fold=5, n_jobs=2):
        print("Begin grid search")
        t0 = time.time()
        kfold = KFold(n_splits=k_fold, shuffle=True)
        if self.kind == "RF":
            parameters = {
                'n_estimators': list(range(50, 251, 10)),
            }
        elif self.kind == "GB":
            parameters = {
                'n_estimators': np.arange(50, 251, 10),
                'learning_rate': np.logspace(-5, 0, 10),
            }
        elif self.kind == "AB":
            parameters = {
                'n_estimators': np.arange(50, 251, 10),
                'learning_rate': np.logspace(-4, 0, 10),
            }

        print(f"Kind: {self.kind}, {self.clf}")
        self.clf = GridSearchCV(self.clf, parameters, scoring='f1', n_jobs=n_jobs, cv=kfold, refit=True)
        self.clf.fit(X, y)
        print(self.clf.cv_results_, '\n', self.clf.best_params_)
        print("Grid_scores: {}".format(self.clf.cv_results_))
        t1 = time.time()
        print("Finish training model")
        print("Elapsed time {:.3f}s".format(t1 - t0))

    def save(self, output_filepath):
        joblib.dump(self, output_filepath)
        print("Classifier saved at {}".format(os.path.abspath(output_filepath)))

    def predict(self, *args, **kwargs):
        return self.clf.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        return self.clf.predict_proba(*args, **kwargs)

    @staticmethod
    def load(classifier_path):
        return joblib.load(classifier_path)


if __name__ == '__main__':
    tp_vcfpath = sys.argv[1]
    fp_vcfpath = sys.argv[2]
    csv_path = sys.argv[3]

    ## vcf to csv
    vcf = VCF2CSV(tp_vcfpath, fp_vcfpath)
    vcf.write_to_csv(csv_path)

    ## load data and preprocess
    vcf_df = VCFDataset(csv_path)
    vcf_df.get_dataset()

    ### train
    clf = Classifier(kind="GB")
    clf.fit(vcf_df.x_train, vcf_df.y_train)

    ## predict
    y_test_pre = clf.predict(vcf_df.x_test)
    y_test_proba = clf.predict_proba(vcf_df.x_test)

    metrics = Metrics(vcf_df.y_test, y_test_pre, y_test_proba[:, 1])
    ## 输出roc 以及相关指标
    metrics.fig_roc()
    print(metrics.header())
    print(metrics.__str__())
