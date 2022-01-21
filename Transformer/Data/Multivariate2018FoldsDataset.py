import torch
from torch.utils.data import Dataset
from Transformer.Data.Multivariate2018FoldsLoader import Multivariate2018FoldsLoader
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class Multivariate2018FoldsDataset(Dataset):

    def __init__(self,
                 fold_index: int,
                 number_of_folds: int,
                 normalization: bool,
                 data_location: str,
                 datasetName: str,
                 train_test: bool,
                 oneHotEncoder: OneHotEncoder,
                 loader: Multivariate2018FoldsLoader):

        if loader is None:
            self.__loader = Multivariate2018FoldsLoader(normalization=normalization,
                                                        data_location=data_location,
                                                        datasetName=datasetName,
                                                        train_test=train_test,
                                                        oneHotEncoder=oneHotEncoder,
                                                        n_splits=number_of_folds)
        else:
            self.__loader = loader
        self.__fold_index = fold_index
        self.__number_of_folds = number_of_folds
        self.__data, self.__labels, self.__percentage = self.__loader.get_fold_by_index(index=fold_index)
        self.__time_length = len(self.__data[0][0])
        self.__channel_length = len(self.__data[0])
        self.__n_classes = len(self.__labels[0])

    def __getitem__(self, index):
        return self.__data[index], self.__labels[index], torch.tensor(np.ones((1, self.__time_length)) > 0, requires_grad=False)

    def __len__(self):
        return len(self.__labels)

    def getOneHotEncoder(self):
        return self.__loader.get_oneHotEncoder()

    def get_channel_length(self):
        return self.__channel_length

    def get_time_length(self):
        return self.__time_length

    def get_percentage(self):
        return self.__percentage

    def new_length(self, new_length: int):
        self.__loader.pad_new_length(final_length=new_length)
        self.__data, self.__labels, self.__percentage = self.__loader.get_fold_by_index(index=self.__fold_index)
        self.__time_length = len(self.__data[0][0])
        self.__channel_length = len(self.__data[0])
        self.__n_classes = len(self.__labels[0])

    def get_number_classes(self):
        return self.__n_classes

    def get_loader(self):
        return self.__loader

    def change_fold(self, index):
        assert index < self.__number_of_folds, "The sub-dataset you want doesn't exist"
        self.__data, self.__labels, self.__percentage = self.__loader.get_fold_by_index(index=index)
