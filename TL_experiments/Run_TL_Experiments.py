from datetime import datetime
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import numpy as np
from Transformer.Data.Multivariate2018Dataset import Multivariate2018Dataset
from Transformer.TransformerTL import TransformerTL
import torch


class Run_TL_Experiments:

    def __init__(self,
                 dataSetName: str,
                 dataLocation: str,
                 normalization: bool,
                 saved_model_path: str):

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
        self.__saved_model_path = saved_model_path

        trainSet_Length = self.__trainSet.get_time_length()
        testSet_length = self.__testSet.get_time_length()
        if trainSet_Length != testSet_length:
            if trainSet_Length > testSet_length:
                self.__testSet.new_length(new_length=trainSet_Length)
            else:
                self.__trainSet.new_length(new_length=testSet_length)

    @staticmethod
    def __buildModel(saved_model_path):
        model = torch.load(saved_model_path)
        return model

    def __runSingleExperiment(self,
                              epochs=100,
                              batch_size=3,
                              learning_rate=1e-4,
                              ReducerLROnPlateau=True,
                              optimizer='Adam',
                              factor=0.1,
                              min_lr=1e-18,
                              patience=10,
                              compute_accuracy_every=5):
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
                                                training_loss_functions_plot=False,
                                                compute_accuracy_every=compute_accuracy_every,
                                                freeze=True)
        return final_accuracy, line

    def __run_Experiments(self,
                          n_exp=5,
                          epochs=4500,
                          batch_size=3,
                          learning_rate=1e-4,
                          ReduceLROnPlateau=False,
                          optimizer='RAdam',
                          factor=0.1,
                          min_lr=1e-18,
                          patience=10,
                          compute_accuracy_every=5):
        accuracy_values = []
        to_print_lines = []
        print("Number of experiments to be performed: ", n_exp, "\n")
        for i in range(n_exp):
            self.__model = self.__buildModel(saved_model_path=self.__saved_model_path)
            print("\nexperiment number: ", i + 1)
            accuracy, line = self.__runSingleExperiment(epochs=epochs,
                                                        batch_size=batch_size,
                                                        learning_rate=learning_rate,
                                                        ReducerLROnPlateau=ReduceLROnPlateau,
                                                        optimizer=optimizer,
                                                        factor=factor,
                                                        min_lr=min_lr,
                                                        patience=patience,
                                                        compute_accuracy_every=compute_accuracy_every)
            accuracy_values.append(accuracy)
            to_print_lines.append(line)
        print("\n\n\n")
        accuracy_values = np.array(accuracy_values)
        mean = np.nanmean(accuracy_values)
        var = np.nanvar(accuracy_values)
        return mean, var, to_print_lines

    def run(self,
            n_exp=5,
            epochs=3500,
            batch_size=64,
            learning_rate=1e-4,
            ReduceLROnPlateau=False,
            optimizer='RAdam',
            factor=0.1,
            min_lr=1e-18,
            patience=10,
            file_to_write_path="",
            source_dataset: str = "",
            compute_accuracy_every=5):
        print_lines = []
        print("Execution started")
        mean, var, lines = self.__run_Experiments(n_exp=n_exp,
                                                  epochs=epochs,
                                                  batch_size=batch_size,
                                                  learning_rate=learning_rate,
                                                  ReduceLROnPlateau=ReduceLROnPlateau,
                                                  optimizer=optimizer,
                                                  factor=factor,
                                                  min_lr=min_lr,
                                                  patience=patience,
                                                  compute_accuracy_every=compute_accuracy_every)
        result_line = "\n\nmean accuracy: " + str(mean) + " variance: " + str(var) + "\n\n"
        print_lines = print_lines + lines
        print_lines.append(result_line)
        txt_file = 'TL_results_source' + "_" + source_dataset + "_target_" + self.__dataSetName + "_" + str(datetime.now().strftime('%Y_%m_%d_%H')) + '.txt'
        txt_file_full = file_to_write_path + "/" + txt_file
        with open(txt_file_full, 'w') as f:
            for line in print_lines:
                f.write(line)
