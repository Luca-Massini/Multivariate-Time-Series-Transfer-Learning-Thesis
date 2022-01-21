from torch.utils.data import Dataset
from Transformer.Data.Multivariate2018Loader import Multivariate2018Loader
from sklearn.preprocessing import OneHotEncoder


class Multivariate2018Dataset(Dataset):
    """
    :param data_location: it is the file path where the archive file is located
    :param normalization: it is a boolean parameter: True if we want to normalize the dataset and False otherwise
    :param dataset_name: Among all the possible dataset in the archive one must be specified by setting this attribute equal to the dataset name
    :param train_test: it is a boolean parameter: True if the training dataset is requested and False if the testing one is requested
    :param oneHotEncoder: it is the OneHotEncoder used to encode the target in one hot encoding format. It can be also equal to None and in that case a new one will be created
    """

    def __init__(self,
                 data_location,
                 normalization,
                 dataset_name: str,
                 train_test: bool,
                 oneHotEncoder: OneHotEncoder):

        self.__oneHotEncoder = oneHotEncoder

        # the object for reading the files and retrieving the dataset in tensor format
        self.__loader = Multivariate2018Loader(normalization=normalization,
                                               data_location=data_location,
                                               datasetName=dataset_name,
                                               train_test=train_test,
                                               oneHotEncoder=oneHotEncoder)
        # get the time series instances and the target classes encoded in one hot encoding format
        self.__time_series_dataset_tensor, self.__targets_tensor, self.__padding_masks = self.__loader.get_dataset()
        # get the length of the time series instances
        self.__time_length = len(self.__time_series_dataset_tensor[0][0])
        # get the number of univariate time series composing each multivariate time series
        self.__channel_length = len(self.__time_series_dataset_tensor[0])

    '''
    It returns the oneHotEncoder object of the loader object if the one provided in the builder method is None and it returns
    the class attribute in the case if it isn't.
    '''

    def getOneHotEncoder(self):
        if self.__oneHotEncoder is None:
            return self.__loader.get_oneHotEncoder()
        else:
            return self.__oneHotEncoder

    def new_length(self, new_length: int):
        self.__loader.pad_new_length(final_length=new_length)
        self.__time_series_dataset_tensor, self.__targets_tensor, self.__padding_masks = self.__loader.get_dataset()
        self.__time_length = len(self.__time_series_dataset_tensor[0][0])
        self.__channel_length = len(self.__time_series_dataset_tensor[0])


    '''
    this method returns the length of the time series contained in the dataset
    '''

    def get_time_length(self):
        return self.__time_length

    '''
    this method returns the number of univariate time series in each multivariate time series
    '''

    def get_channel_length(self):
        return self.__channel_length

    '''
    this method returns the number of classes in the classification problem
    '''

    def get_number_classes(self):
        return len(self.__targets_tensor[0])

    '''
    it returns a tuple containing the idx-th time series in the dataset and its target class.
    Both the time series and the target class are in tensor format.
    The target class is encoded in one hot encoding format
    '''

    def __getitem__(self, index):
        return self.__time_series_dataset_tensor[index], self.__targets_tensor[index], self.__padding_masks[index]

    '''
    it returns the number of time series in the dataset
    '''

    def __len__(self):
        return len(self.__targets_tensor)
