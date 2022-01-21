from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split

from Transformer.Data.Multivariate2018Loader import Multivariate2018Loader
from math import floor
import torch


class Multivariate2018FoldsLoader(Multivariate2018Loader):
    def __init__(self,
                 normalization: bool,
                 data_location,
                 datasetName: str,
                 train_test: bool,
                 oneHotEncoder: OneHotEncoder,
                 n_splits: int):
        super().__init__(normalization, data_location, datasetName, train_test, oneHotEncoder)
        self.__n_splits = n_splits
        self.__data, self.__labels, _ = self.get_dataset()
        self.__data = self.__data.numpy()
        self.__data, self.__labels, self.__percentages = self.__get_folds()

    def __get_folds(self):
        ts_splits_data_vector, ts_splits_label_array = [self.__data], [self.__labels]
        number_of_iterations = floor(self.__n_splits / 2) + 1
        for _ in range(number_of_iterations):
            data = ts_splits_data_vector[-1]
            labels = ts_splits_label_array[-1]
            x_train, x_test, y_train, y_test = train_test_split(data,
                                                                labels,
                                                                test_size=0.5,
                                                                random_state=0,
                                                                shuffle=True,
                                                                stratify=labels)
            ts_splits_data_vector.append(x_train)
            ts_splits_label_array.append(y_train)
        ts_splits_data_vector, ts_splits_label_array = list(reversed(ts_splits_data_vector)), list(
            reversed(ts_splits_label_array))
        number_of_series = len(ts_splits_data_vector[-1])
        percentage_values = [len(dataset) / number_of_series * 100 for dataset in ts_splits_data_vector]
        ts_splits_data_vector = [torch.tensor(ts_dataset) for ts_dataset in ts_splits_data_vector]
        ts_splits_label_array = [torch.tensor(target_dataset) for target_dataset in ts_splits_label_array]
        return ts_splits_data_vector, ts_splits_label_array, percentage_values

    def pad_new_length(self, final_length: int):
        super(Multivariate2018FoldsLoader, self).pad_new_length(final_length=final_length)
        data_tensors, labels, _ = self.get_dataset()
        self.__data = data_tensors.numpy()
        self.__labels = np.array([np.argmax(row) for row in labels])
        self.__data, self.__labels, self.__percentages = self.__get_folds()

    def get_fold_by_index(self, index):
        assert index < self.__n_splits, "The fold you want doesn't exist since there's no the index-th fold"
        return self.__data[index], self.__labels[index], self.__percentages[index]
