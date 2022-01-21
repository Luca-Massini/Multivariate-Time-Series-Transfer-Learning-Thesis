from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import torch
import numpy as np


class Multivariate2018Loader:
    """
    :param normalization: it is a boolean parameter: True if we want to normalize the dataset and False otherwise
    :param data_location: it is the file path where the archive file is located
    :param datasetName: Among all the possible dataset in the archive one must be specified by setting this attribute equal to the dataset name
    :param train_test: it is a boolean parameter: True if the train dataset is requested and False if the tested one is requested
    :param oneHotEncoder: it is the OneHotEncoder used to encode the target in one hot encoding format. It can be also equal to null
    """

    def __init__(self,
                 normalization: bool,
                 data_location,
                 datasetName: str,
                 train_test: bool,
                 oneHotEncoder: OneHotEncoder):

        self.__oneHotEncoder = oneHotEncoder

        # if the path to the folder where the archive is saved is not provided the default path is used
        if data_location is None:
            self.__DATA_LOCATION = '../Data/Multivariate_ts/Multivariate_ts/'
        else:
            self.__DATA_LOCATION = data_location

        self.__normalization = normalization

        # get the dataset as a tuple (time series instances, target classes)
        self.__data, self.__labels, self.__padding_masks = self.__get_data_and_labels(datasetName=datasetName,
                                                                                      train_test=train_test)
        self.__channel_number = self.__data.shape[1]

    def __get_data_and_labels(self, datasetName: str, train_test: bool):
        if train_test:
            train_test_string = 'TRAIN'
        else:
            train_test_string = 'TEST'
        filepath = self.__DATA_LOCATION + '/' + datasetName + '/' + datasetName + '_' + train_test_string + '.ts'
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                   replace_missing_vals_with='NaN')

        if self.__oneHotEncoder is None:
            self.__oneHotEncoder = self.__get_new_oneHotEncoder(labels=labels)
            labels = np.array(labels).reshape((len(labels), 1))
            labels = self.__oneHotEncoder.transform(labels).toarray()
        else:
            labels = np.array(labels).reshape((len(labels), 1))
            labels = self.__oneHotEncoder.transform(labels).toarray()

        indices = df.index.array
        lengths = []
        for index in indices:
            lengths.append(len(df['dim_0'].loc[index]))
        padding_masks = self.__compute_padding_masks(lengths=lengths, number_of_time_series=len(indices))

        lengths = df.applymap(lambda x: len(x)).values
        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)
        if self.__normalization:
            df = (df - df.mean()) / (df.std() + np.finfo(float).eps)
        data_tensors = []
        for index in indices:
            data_tensors.append(torch.tensor(df.loc[index].values).transpose(-1, -2))
        max_length = np.max(lengths)
        data_tensors = [self.__get_new_length_tensor(ts_tensor=tensor, length_to_be_added=max_length - tensor.shape[1])
                        for tensor in data_tensors]
        data_tensors = torch.stack(data_tensors)
        return data_tensors, labels, padding_masks

    def pad_new_length(self, final_length: int):
        assert(final_length > self.__data.shape[2])
        result_data = []
        result_masks = []
        for ts_tensor in self.__data:
            result_data.append(self.__get_new_length_tensor(ts_tensor=ts_tensor,
                                                            length_to_be_added=final_length - ts_tensor.shape[1]))
        self.__data = torch.stack(result_data)
        for mask in self.__padding_masks:
            np_mask = mask.numpy()
            added_padding = np.full(shape=(final_length - np_mask.shape[0]), fill_value=False)
            result_masks.append(np.append(np_mask, added_padding))
        self.__padding_masks = torch.tensor(result_masks)

    @staticmethod
    def __get_new_length_tensor(ts_tensor: torch.tensor, length_to_be_added: int):
        np_tensor = ts_tensor.numpy()
        np_tensor = np.pad(array=np_tensor, pad_width=((0, 0), (0, length_to_be_added)))
        new_ts_tensor = torch.tensor(np_tensor)
        return new_ts_tensor

    @staticmethod
    def __get_new_oneHotEncoder(labels):
        oneHotEncoder = OneHotEncoder()
        labels_ = np.array(labels).reshape((len(labels), 1))
        oneHotEncoder.fit(labels_)
        return oneHotEncoder

    @staticmethod
    def __compute_padding_masks(lengths: list, number_of_time_series: int):
        max_length = np.max(lengths)
        padding_masks = []
        for index in range(number_of_time_series):
            length = lengths[index]
            non_padding_mask = np.full((length,), True)
            padding_length = max_length - length
            if padding_length > 0:
                padding_mask = np.full((padding_length,), False)
                padding_masks.append(np.append(non_padding_mask, padding_mask))
            else:
                padding_masks.append(non_padding_mask)
        padding_masks = torch.tensor(padding_masks)
        return padding_masks

    '''
    get the oneHotEncoder object used to encode the target classes in one hot encoding format
    '''

    def get_oneHotEncoder(self):
        return self.__oneHotEncoder

    '''
    get the dataset in tuple format (time series instances, target classes). Both the time series and the target classes are
    in tensor format and the target classes are encoded in one hot encoding format
    '''

    def get_dataset(self):
        return self.__data, self.__labels, self.__padding_masks

    '''
    given the dataset name, this method returns the number of time series that compose each multivariate time series
    in the dataset
    '''

    def __get_channels_number(self):
        return self.__channel_number
