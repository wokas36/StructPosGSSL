import numpy as np
import torch
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import DataLoader
import pdb

class EmbeddingEvaluation():
    def __init__(self, base_classifier, evaluator, task_type, num_tasks, device, params_dict=None, param_search=True,is_rand_label=False):
        self.is_rand_label = is_rand_label
        self.base_classifier = base_classifier
        self.evaluator = evaluator
        self.eval_metric = evaluator.eval_metric
        self.task_type = task_type
        self.num_tasks = num_tasks
        self.device = device
        self.param_search = param_search
        self.params_dict = params_dict
        if self.eval_metric == 'rmse':
            self.gscv_scoring_name = 'neg_root_mean_squared_error'
        elif self.eval_metric == 'mae':
            self.gscv_scoring_name = 'neg_mean_absolute_error'
        elif self.eval_metric == 'rocauc':
            self.gscv_scoring_name = 'roc_auc'
        elif self.eval_metric == 'accuracy':
            self.gscv_scoring_name = 'accuracy'
        else:
            raise ValueError('Undefined grid search scoring for metric %s ' % self.eval_metric)

        self.classifier = None
    def scorer(self, y_true, y_raw):
        input_dict = {"y_true": y_true, "y_pred": y_raw}
        score = self.evaluator.eval(input_dict)[self.eval_metric]
        return score

    def ee_binary_classification(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):
        if self.param_search:
            params_dict = {'C': [0.001, 0.01,0.1,1,10,100,1000]}
            self.classifier = make_pipeline(StandardScaler(),
                                            GridSearchCV(self.base_classifier, params_dict, cv=5, 
                                                         scoring=self.gscv_scoring_name, n_jobs=16, verbose=0)
                                            )
        else:
            self.classifier = make_pipeline(StandardScaler(), self.base_classifier)


        self.classifier.fit(train_emb, np.squeeze(train_y))

        if self.eval_metric == 'accuracy':
            train_raw = self.classifier.predict(train_emb)
            val_raw = self.classifier.predict(val_emb)
            test_raw = self.classifier.predict(test_emb)
        else:
            train_raw = self.classifier.predict_proba(train_emb)[:, 1]
            val_raw = self.classifier.predict_proba(val_emb)[:, 1]
            test_raw = self.classifier.predict_proba(test_emb)[:, 1]

        return np.expand_dims(train_raw, axis=1), np.expand_dims(val_raw, axis=1), np.expand_dims(test_raw, axis=1)

    def ee_multioutput_binary_classification(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):

        params_dict = {
            'multioutputclassifier__estimator__C': [1e-1, 1e0, 1e1, 1e2]}
        self.classifier = make_pipeline(StandardScaler(), MultiOutputClassifier(
            self.base_classifier, n_jobs=-1))

        if np.isnan(train_y).any():
            print("Has NaNs ... ignoring them")
            train_y = np.nan_to_num(train_y)
        self.classifier.fit(train_emb, train_y)

        train_raw = np.transpose([y_pred[:, 1] for y_pred in self.classifier.predict_proba(train_emb)])
        val_raw = np.transpose([y_pred[:, 1] for y_pred in self.classifier.predict_proba(val_emb)])
        test_raw = np.transpose([y_pred[:, 1] for y_pred in self.classifier.predict_proba(test_emb)])

        return train_raw, val_raw, test_raw

    def ee_regression(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):
        if self.param_search:
            params_dict = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]}
            self.classifier = GridSearchCV(self.base_classifier, params_dict, cv=5,
                                      scoring=self.gscv_scoring_name, n_jobs=16, verbose=0)
        else:
            self.classifier = self.base_classifier

        self.classifier.fit(train_emb, np.squeeze(train_y))

        train_raw = self.classifier.predict(train_emb)
        val_raw = self.classifier.predict(val_emb)
        test_raw = self.classifier.predict(test_emb)

        return np.expand_dims(train_raw, axis=1), np.expand_dims(val_raw, axis=1), np.expand_dims(test_raw, axis=1)

    def embedding_evaluation(self, encoder, train_emb, train_y, val_emb, val_y, test_emb, test_y):
        if 'classification' in self.task_type:
            if self.num_tasks == 1:
                train_raw, val_raw, test_raw = self.ee_binary_classification(train_emb, train_y, val_emb, val_y, test_emb,
                                                                        test_y)
            elif self.num_tasks > 1:
                train_raw, val_raw, test_raw = self.ee_multioutput_binary_classification(train_emb, train_y, val_emb, val_y,
                                                                                    test_emb, test_y)
            else:
                raise NotImplementedError
        else:
            if self.num_tasks == 1:
                train_raw, val_raw, test_raw = self.ee_regression(train_emb, train_y, val_emb, val_y, test_emb, test_y)
            else:
                raise NotImplementedError

        train_score = self.scorer(train_y, train_raw)

        val_score = self.scorer(val_y, val_raw)

        test_score = self.scorer(test_y, test_raw)

        return train_score, val_score, test_score