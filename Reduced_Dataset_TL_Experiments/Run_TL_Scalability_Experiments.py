from Transformer.Data.Multivariate2018Dataset import Multivariate2018Dataset
from Transformer.TransformerTL import TransformerTL
from Transformer.Data.Multivariate2018FoldsDataset import Multivariate2018FoldsDataset
import torch
import numpy as np
from datetime import datetime


class Run_TL_Scalability_Experiments:

    def __init__(self,
                 target_dataset_train: Multivariate2018FoldsDataset,
                 target_dataset_test: Multivariate2018Dataset,
                 saved_Tl_Model_Path: str,
                 number_of_subsets: int,
                 datasetName: str):
        self.__datasetName = datasetName
        self.__target_dataset_train = target_dataset_train
        self.__target_dataset_test = target_dataset_test
        self.__saved_Tl_Model_Path = saved_Tl_Model_Path
        self.__n_splits = number_of_subsets
        self.__from_scratch_model = None
        self.__filter_size_autoEncoder = 3,
        self.__n_layers_autoEncoder = 1,
        self.__convolutions_per_layer_autoEncoder = 1,
        self.__n_filters_index_autoEncoder = 2,
        self.__autoEncoder_encoding_channel_size = 10,
        self.__reducing_length_factor_autoEncoder = 2,
        self.__embedding_size = 256,
        self.__time_series_channels_number = self.__target_dataset_train.get_channel_length(),
        self.__feedForward_dimension = 256,
        self.__nHeads = 16,
        self.__dropout = 0.1,
        self.__time_series_length = self.__target_dataset_train.get_time_length(),
        self.__n_classes = self.__target_dataset_train.get_number_classes(),
        self.__n_encoders = 3,
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_model_hyper_parameters(self,
                                   filter_size_autoEncoder=3,
                                   n_layers_autoEncoder=1,
                                   convolutions_per_layer_autoEncoder=1,
                                   n_filters_index_autoEncoder=2,
                                   autoEncoder_encoding_channel_size=10,
                                   reducing_length_factor_autoEncoder=2,
                                   embedding_size=256,
                                   feedForward_dimension=256,
                                   nHeads=16,
                                   dropout=0.1,
                                   n_encoders=3,
                                   device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.__filter_size_autoEncoder = filter_size_autoEncoder,
        self.__n_layers_autoEncoder = n_layers_autoEncoder,
        self.__convolutions_per_layer_autoEncoder = convolutions_per_layer_autoEncoder,
        self.__n_filters_index_autoEncoder = n_filters_index_autoEncoder,
        self.__autoEncoder_encoding_channel_size = autoEncoder_encoding_channel_size,
        self.__reducing_length_factor_autoEncoder = reducing_length_factor_autoEncoder,
        self.__embedding_size = embedding_size,
        self.__feedForward_dimension = feedForward_dimension,
        self.__nHeads = nHeads,
        self.__dropout = dropout,
        self.__n_encoders = n_encoders,
        self.__device = device

    def build_from_scratch_model(self):
        self.__from_scratch_model = TransformerTL(filter_size_autoEncoder=self.__filter_size_autoEncoder,
                                                  n_layers_autoEncoder=self.__n_layers_autoEncoder,
                                                  convolutions_per_layer_autoEncoder=self.__convolutions_per_layer_autoEncoder,
                                                  n_filters_index_autoEncoder=self.__n_filters_index_autoEncoder,
                                                  autoEncoder_encoding_channel_size=self.__autoEncoder_encoding_channel_size,
                                                  reducing_length_factor_autoEncoder=self.__reducing_length_factor_autoEncoder,
                                                  embedding_size=self.__embedding_size,
                                                  time_series_channels_number=self.__time_series_channels_number,
                                                  feedForward_dimension=self.__feedForward_dimension,
                                                  nHeads=self.__nHeads,
                                                  dropout=self.__dropout,
                                                  time_series_length=self.__time_series_length,
                                                  n_classes=self.__n_classes,
                                                  n_encoders=self.__n_encoders,
                                                  device=self.__device)

    def run_fold_experiment_TL(self,
                               batchSize=64,
                               epochs=4500,
                               learning_rate=1e-4,
                               ReduceLROnPlateau=False,
                               optimizer='RAdam',
                               factor=0.1,
                               min_lr=1e-18,
                               patience=25,
                               print_accuracy_test=False,
                               verbose=False,
                               weight_decay=0,
                               training_loss_functions_plot=False,
                               freeze=True,
                               compute_accuracy_every=5,
                               fold_index: int = 0,
                               n_experiments: int = 5,
                               file_to_write_path=''):
        accuracy_values = []
        file_lines = []
        assert fold_index <= self.__n_splits - 1, "this dataset subset does not exist"
        self.__target_dataset_train.change_fold(fold_index)
        percentage = self.__target_dataset_train.get_percentage()
        line = "Percentage of samples included in this subset of the training one: ", percentage, "%"
        file_lines.append(line)
        print(line)
        for _ in range(n_experiments):
            tl_model = torch.load(self.__saved_Tl_Model_Path)
            tl_model = tl_model.get_transfer_learning_model(new_ts_length=self.__target_dataset_train.get_time_length(),
                                                            new_number_of_variables=self.__target_dataset_train.get_channel_length(),
                                                            new_n_classes=self.__target_dataset_train.get_number_classes())
            accuracy, line_to_be_printed = tl_model.fit(trainingSet=self.__target_dataset_train,
                                                        batchSize=batchSize,
                                                        epochs=epochs,
                                                        learning_rate=learning_rate,
                                                        ReduceLROnPlateau=ReduceLROnPlateau,
                                                        optimizer=optimizer,
                                                        factor=factor,
                                                        min_lr=min_lr,
                                                        patience=patience,
                                                        validationDataset=self.__target_dataset_test,
                                                        print_accuracy_test=print_accuracy_test,
                                                        verbose=verbose,
                                                        weight_decay=weight_decay,
                                                        training_loss_functions_plot=training_loss_functions_plot,
                                                        freeze=freeze,
                                                        compute_accuracy_every=compute_accuracy_every)
            print(line_to_be_printed)
            file_lines.append(line_to_be_printed)
            accuracy_values.append(accuracy)
            print("\n\n")
        mean_accuracy = np.mean(accuracy_values)
        var = np.nanvar(accuracy_values)
        line = "the mean accuracy is: " + str(mean_accuracy) + " and the variance is: " + str(var)
        print(line)
        file_lines.append(line)
        txt_file = 'results_TL_reduced_dimension' + "_" + self.__datasetName + "_" + str(
            datetime.now().strftime('%Y_%m_%d_%H')) + '.txt'
        txt_file_full = file_to_write_path + "/" + txt_file
        with open(txt_file_full, 'w') as f:
            for line in file_lines:
                f.write(line)

    def run_fold_experiments_scratch(self,
                                     batchSize=64,
                                     epochs=4500,
                                     learning_rate=1e-4,
                                     ReduceLROnPlateau=False,
                                     optimizer='RAdam',
                                     factor=0.1,
                                     min_lr=1e-18,
                                     patience=25,
                                     print_accuracy_test=False,
                                     verbose=False,
                                     weight_decay=0,
                                     training_loss_functions_plot=False,
                                     freeze=False,
                                     compute_accuracy_every=5,
                                     fold_index: int = 0,
                                     n_experiments: int = 5,
                                     file_to_write_path=''):
        assert fold_index <= self.__n_splits - 1, "this dataset subset does not exist"
        assert self.__from_scratch_model is not None, "You did not build the model before executing the experiment so it is None"
        scratch_accuracy_values = []
        file_lines = []
        line = "Training from scratch using reducing the training set size \n"
        file_lines.append(line)
        print(line)
        self.__target_dataset_train.change_fold(fold_index)
        percentage = self.__target_dataset_train.get_percentage()
        line = "Percentage of samples included in this subset of the training one: ", percentage, "%"
        file_lines.append(line)
        print(line)
        for _ in range(n_experiments):
            self.build_from_scratch_model()
            best_validation_accuracy, line_to_be_printed = self.__from_scratch_model.fit(
                trainingSet=self.__target_dataset_train,
                batchSize=batchSize,
                epochs=epochs,
                learning_rate=learning_rate,
                ReduceLROnPlateau=ReduceLROnPlateau,
                optimizer=optimizer,
                factor=factor,
                min_lr=min_lr,
                patience=patience,
                validationDataset=self.__target_dataset_test,
                print_accuracy_test=print_accuracy_test,
                verbose=verbose,
                weight_decay=weight_decay,
                training_loss_functions_plot=training_loss_functions_plot,
                freeze=freeze,
                compute_accuracy_every=compute_accuracy_every)
            file_lines.append(line_to_be_printed)
            print(line_to_be_printed)
            scratch_accuracy_values.append(best_validation_accuracy)
            print("\n\n")
        mean_accuracy = np.mean(scratch_accuracy_values)
        var = np.nanvar(scratch_accuracy_values)
        line = "the mean accuracy is: " + str(mean_accuracy) + " and the variance is: " + str(var)
        print(line)
        file_lines.append(line)
        txt_file = 'results_scratch_reduced_dimension' + "_" + self.__datasetName + "_" + str(
            datetime.now().strftime('%Y_%m_%d_%H')) + '.txt'
        txt_file_full = file_to_write_path + "/" + txt_file
        with open(txt_file_full, 'w') as f:
            for line in file_lines:
                f.write(line)
