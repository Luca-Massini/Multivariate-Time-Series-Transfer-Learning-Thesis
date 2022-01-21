import math

import torch.nn as nn
import torch
from torch.nn import Module


class TransformerTL_Loss(Module):
    def __init__(self, verbose: bool):
        super(TransformerTL_Loss, self).__init__()
        self.__classification_loss = nn.CrossEntropyLoss()
        self.__reconstruction_loss = nn.MSELoss()
        self.log_vars = nn.Parameter(torch.tensor([1.0, 1.0]))
        self.__verbose = verbose
        self.__classification_loss_last_value = None
        self.__time_series_reconstruction_loss_value = None
        self.__loss_value = None

    def forward(self, true_class, predicted_class, reconstructed_time_series, original_time_series):
        true_classification_loss = self.__classification_loss(predicted_class, true_class)
        self.__classification_loss_last_value = true_classification_loss.tolist()
        if self.__verbose:
            print("\n")
            print("-classification training loss value: ", true_classification_loss.tolist())
        classification_loss = true_classification_loss * (1 / (self.log_vars[0] ** 2)) + math.log(self.log_vars[0])
        true_reconstruction_loss = self.__reconstruction_loss(reconstructed_time_series, original_time_series)
        self.__time_series_reconstruction_loss_value = true_reconstruction_loss.tolist()
        if self.__verbose:
            print(" reconstruction training loss value: ", self.__time_series_reconstruction_loss_value)
        encoding_loss = true_reconstruction_loss * (1/(2 * (self.log_vars[1] ** 2))) + math.log(self.log_vars[1])
        loss_value = classification_loss + encoding_loss
        self.__loss_value = loss_value.tolist()
        return loss_value

    def get_last_classification_loss_value(self):
        return self.__classification_loss_last_value

    def get_last_reconstruction_loss_value(self):
        return self.__time_series_reconstruction_loss_value

    def get_last_loss_value(self):
        return self.__loss_value
