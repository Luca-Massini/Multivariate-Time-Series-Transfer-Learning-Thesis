from math import floor
import torch.nn.functional as functional
import torch.nn as nn


class ConvolutionalEncoder(nn.Module):

    def __init__(self,
                 channel_length,
                 filter_size=3,
                 n_layers=4,
                 convolutions_per_layer=1,
                 n_filters_index=2,
                 final_number_filters=5,
                 down_sampling_factor=2,
                 dropout=0.5):
        super().__init__()

        assert (filter_size % 2 != 0 and filter_size > 0)
        assert (n_layers > 0)
        assert (convolutions_per_layer > 0)
        assert (n_filters_index > 1)
        assert (channel_length >= 1)
        assert (down_sampling_factor % 2 == 0 or down_sampling_factor == 1)

        self.__down_sampling_factor = down_sampling_factor

        self.__n_filters_index = n_filters_index

        self.__padding = None

        n_filters = final_number_filters * (n_filters_index ** (n_layers - 1))

        self.__initial_number_of_filters = n_filters
        self.__convolutions_per_layer = convolutions_per_layer
        self.__filter_size = filter_size

        self.__modules = []

        in_channels = channel_length
        for layer in range(n_layers):
            convolutions = []
            out_channels = n_filters
            convolution_number = 0
            for convolutional_layer in range(convolutions_per_layer):
                if convolution_number == convolutions_per_layer - 1:
                    stride = (down_sampling_factor,)
                else:
                    stride = (1,)
                convolutions.append(nn.Conv1d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=(filter_size,),
                                              stride=stride,
                                              padding=(floor(filter_size / 2),)
                                              ))
                in_channels = out_channels
                convolution_number += 1
            self.__modules += convolutions
            if layer != (n_layers - 1):
                self.__modules += [nn.Dropout(p=dropout)]
            self.__modules += [nn.LeakyReLU()]
            in_channels = n_filters
            n_filters = n_filters // n_filters_index

        self.__moduleList = nn.ModuleList(self.__modules)

        self.__n_layers = n_layers
        self.__last_channel_size_before_encoding = in_channels

    def get_last_channel_size_before_encoding(self):
        return self.__last_channel_size_before_encoding

    @staticmethod
    def __compute_padding(time_series, down_sampling_factor, n_layers):
        length = time_series.shape[2]
        reducing_factor = down_sampling_factor ** n_layers
        if length % reducing_factor != 0:
            length_after_encoding = floor(length // reducing_factor)
            new_length = (length_after_encoding + 1) * reducing_factor
            length_difference = new_length - length
            first_pad = length_difference // 2
            second_pad = length_difference - first_pad
            padding = (first_pad, second_pad)
        else:
            padding = (0, 0)
        return padding

    def forward(self, x):
        if self.__padding is None:
            padding = self.__compute_padding(time_series=x, down_sampling_factor=self.__down_sampling_factor,
                                             n_layers=self.__n_layers)
            self.__padding = padding
        else:
            padding = self.__padding
        x = functional.pad(input=x, pad=padding, mode='constant', value=0)
        for module in self.__moduleList:
            x = module(x)
        return x, padding

    def adapt_to_new_Dataset(self, new_number_Of_Variables):
        convolutions = []
        in_channels = new_number_Of_Variables
        out_channels = self.__initial_number_of_filters
        convolution_number = 0
        for convolutional_layer in range(self.__convolutions_per_layer):
            if convolution_number == self.__convolutions_per_layer - 1:
                stride = (self.__down_sampling_factor,)
            else:
                stride = (1,)
            convolutions.append(nn.Conv1d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=(self.__filter_size,),
                                          stride=stride,
                                          padding=(floor(self.__filter_size / 2),)
                                          ))
            in_channels = out_channels
            convolution_number += 1
        for i in range(self.__convolutions_per_layer):
            self.__modules[i] = convolutions[i]
        self.__moduleList = nn.ModuleList(self.__modules)
        self.__padding = None

