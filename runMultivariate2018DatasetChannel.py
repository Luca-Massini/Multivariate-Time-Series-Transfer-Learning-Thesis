from datetime import datetime

import numpy as np
from Transformer.Data.Multivariate2018Dataset import Multivariate2018Dataset
from Transformer.TransformerTL import TransformerTL
import torch


class runMultivariate2018DatasetChannel:

    def __init__(self,
                 filter_size_autoEncoder: int,
                 n_layers_autoEncoder: int,
                 convolutions_per_layer_autoEncoder: int,
                 n_filters_index_autoEncoder: int,
                 reducing_length_factor_autoEncoder: int,
                 embedding_size: int,
                 feedForward_dimension: int,
                 nHeads: int,
                 dropout: float,
                 n_encoders: int,
                 device: torch.device,
                 dataSetName: str,
                 dataLocation: str,
                 normalization: bool):

        self.__model = None
        self.__dataSetName = dataSetName
        self.__trainSet = Multivariate2018Dataset(data_location=dataLocation,
                                                  normalization=normalization,
                                                  dataset_name=dataSetName,
                                                  train_test=True,
                                                  oneHotEncoder=None)

        self.__testSet = Multivariate2018Dataset(data_location=dataLocation,
                                                 normalization=normalization,
                                                 dataset_name=dataSetName,
                                                 train_test=False,
                                                 oneHotEncoder=self.__trainSet.getOneHotEncoder())

        trainSet_Length = self.__trainSet.get_time_length()
        testSet_length = self.__testSet.get_time_length()
        if trainSet_Length != testSet_length:
            if trainSet_Length > testSet_length:
                self.__testSet.new_length(new_length=trainSet_Length)
            else:
                self.__trainSet.new_length(new_length=testSet_length)

        self.__device = device
        self.__embedding_size = embedding_size
        self.__n_encoders = n_encoders
        self.__nHeads = nHeads
        self.__dropout = dropout
        self.__filter_size_autoEncoder = filter_size_autoEncoder
        self.__n_layers_autoEncoder = n_layers_autoEncoder
        self.__convolutions_per_layer_autoEncoder = convolutions_per_layer_autoEncoder
        self.__n_filter_index_autoEncoder = n_filters_index_autoEncoder
        self.__reducing_length_factor_autoEncoder = reducing_length_factor_autoEncoder
        self.__timeSeries_length = self.__trainSet.get_time_length()
        self.__n_classes = self.__trainSet.get_number_classes()
        self.__time_series_channels_number = self.__trainSet.get_channel_length()
        self.__feed_forward_dimension = feedForward_dimension

    def __buildModel(self,
                     encoding_channel_size):
        model = TransformerTL(filter_size_autoEncoder=self.__filter_size_autoEncoder,
                              n_layers_autoEncoder=self.__n_layers_autoEncoder,
                              convolutions_per_layer_autoEncoder=self.__convolutions_per_layer_autoEncoder,
                              n_filters_index_autoEncoder=self.__n_filter_index_autoEncoder,
                              autoEncoder_encoding_channel_size=encoding_channel_size,
                              reducing_length_factor_autoEncoder=self.__reducing_length_factor_autoEncoder,
                              embedding_size=self.__embedding_size,
                              time_series_channels_number=self.__time_series_channels_number,
                              feedForward_dimension=self.__feed_forward_dimension,
                              nHeads=self.__nHeads,
                              dropout=self.__dropout,
                              time_series_length=self.__timeSeries_length,
                              n_classes=self.__n_classes,
                              n_encoders=self.__n_encoders,
                              device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")).to(self.__device)
        return model

    def __runSingleExperiment(self,
                              epochs=100,
                              batch_size=3,
                              learning_rate=1e-4,
                              ReducerLROnPlateau=True,
                              optimizer='Adam',
                              factor=0.1,
                              min_lr=1e-18,
                              patience=10):
        final_accuracy, line = self.__model.fit(trainingSet=self.__trainSet,
                                                batchSize=batch_size,
                                                epochs=epochs,
                                                learning_rate=learning_rate,
                                                ReduceLROnPlateau=ReducerLROnPlateau,
                                                optimizer=optimizer,
                                                factor=factor,
                                                min_lr=min_lr,
                                                patience=patience,
                                                validationDataset=self.__testSet,
                                                print_accuracy_test=False,
                                                verbose=False,
                                                weight_decay=0,
                                                training_loss_functions_plot=False)
        return final_accuracy, line

    def __run_channel_Experiments(self,
                                  n_exp=5,
                                  epochs=4500,
                                  batch_size=3,
                                  learning_rate=1e-4,
                                  ReduceLROnPlateau=False,
                                  optimizer='RAdam',
                                  factor=0.1,
                                  min_lr=1e-18,
                                  patience=10,
                                  channel_size=10):
        accuracy_values = []
        to_print_lines = []
        print("AutoEncoder channel size: ", channel_size,
              "\nNumber of experiments to be performed: ", n_exp)
        for i in range(n_exp):
            self.__model = self.__buildModel(encoding_channel_size=channel_size)
            print("\nexperiment number: ", i + 1)
            accuracy, line = self.__runSingleExperiment(epochs=epochs,
                                                        batch_size=batch_size,
                                                        learning_rate=learning_rate,
                                                        ReducerLROnPlateau=ReduceLROnPlateau,
                                                        optimizer=optimizer,
                                                        factor=factor,
                                                        min_lr=min_lr,
                                                        patience=patience)
            accuracy_values.append(accuracy)
            to_print_lines.append(line)
        print("\n\n\n")
        accuracy_values = np.array(accuracy_values)
        mean = np.nanmean(accuracy_values)
        var = np.nanvar(accuracy_values)
        return mean, var, to_print_lines

    def run(self,
            n_exp=5,
            epochs=4500,
            batch_size=64,
            learning_rate=1e-4,
            ReduceLROnPlateau=False,
            optimizer='RAdam',
            factor=0.1,
            min_lr=1e-18,
            patience=10,
            channel_sizes=[3, 5, 10],
            file_to_write_path=""):
        print_lines = []
        print("Execution started")
        for channel_size in channel_sizes:
            mean, var, lines = self.__run_channel_Experiments(n_exp=n_exp,
                                                              epochs=epochs,
                                                              batch_size=batch_size,
                                                              learning_rate=learning_rate,
                                                              ReduceLROnPlateau=ReduceLROnPlateau,
                                                              optimizer=optimizer,
                                                              factor=factor,
                                                              min_lr=min_lr,
                                                              patience=patience,
                                                              channel_size=channel_size)
            result_line = "\n\nmean accuracy: " + str(mean) + " variance: " + str(var) + "\n\n"
            print_lines.append("AutoEncoder channel size: " + str(channel_size))
            print_lines = print_lines + lines
            print_lines.append(result_line)
        txt_file = 'results' + "_" + self.__dataSetName + "_" + str(datetime.now().strftime('%Y_%m_%d_%H')) + '.txt'
        txt_file_full = file_to_write_path + "/" + txt_file
        with open(txt_file_full, 'w') as f:
            for line in print_lines:
                f.write(line)
