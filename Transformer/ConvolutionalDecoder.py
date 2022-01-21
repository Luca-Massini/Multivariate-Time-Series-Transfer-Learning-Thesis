from math import floor
import torch.nn as nn


class ConvolutionalDecoder(nn.Module):

    def __init__(self,
                 input_channel_size=3,
                 filter_size=3,
                 n_layers=4,
                 convolutions_per_layer=1,
                 n_filters_index=2,
                 before_encoding_channel_size=5,
                 original_channel_length=3,
                 upSampling_factor=2,
                 dropout=0.5):

        super().__init__()

        assert (filter_size % 2 != 0 and filter_size > 0)
        assert (n_layers > 0)
        assert (convolutions_per_layer > 0)
        assert (n_filters_index > 1)
        assert (before_encoding_channel_size > 0)
        assert (input_channel_size > 0)
        assert (original_channel_length > 0)

        self.__filter_size = filter_size
        self.__convolutions_per_layer = convolutions_per_layer
        self.__n_layers = n_layers
        self.__n_filter_index = n_filters_index

        self.__upSampling_factor = upSampling_factor

        self.__padding = None

        self.__modules = []

        if n_layers > 1:
            n_filters = before_encoding_channel_size * n_filters_index
        else:
            n_filters = original_channel_length

        self.__before_encoding_channel_size = before_encoding_channel_size

        self.__in_channels = before_encoding_channel_size
        for layer in range(n_layers):
            self.__out_channels = n_filters
            first_convolution_boolean = True
            self.__padding = floor(filter_size / 2)
            for convolutional_layer in range(convolutions_per_layer):
                if first_convolution_boolean:
                    self.__stride_tuple = (upSampling_factor,)
                    self.__output_padding = (upSampling_factor - self.__padding,)
                    first_convolution_boolean = False
                else:
                    self.__stride_tuple = (1,)
                    self.__output_padding = (0,)
                self.__modules.append(nn.ConvTranspose1d(in_channels=int(self.__in_channels),
                                                         out_channels=int(self.__out_channels),
                                                         kernel_size=(int(filter_size),),
                                                         stride=self.__stride_tuple,
                                                         padding=(self.__padding,),
                                                         output_padding=self.__output_padding
                                                         ))
                self.__in_channels = self.__out_channels
            if layer != (n_layers - 1):
                self.__modules += [nn.Dropout(p=dropout)]
            self.__modules += [nn.LeakyReLU()]
            self.__in_channels = n_filters
            if layer != (n_layers - 2):
                n_filters = n_filters * n_filters_index
            else:
                n_filters = original_channel_length
        self.__moduleList = nn.ModuleList(self.__modules)

    def forward(self, x, added_padding=(0, 0)):
        for module in self.__moduleList:
            x = module(x)
        x = x[:, :, added_padding[0]: (x.shape[2] - added_padding[1])]
        return x

    def adapt_to_new_Dataset(self, new_number_Of_Variables):
        first_convolution_boolean = True
        convolutions = []
        in_channels = self.__before_encoding_channel_size * (self.__n_filter_index ** (self.__n_layers-1))
        out_channels = new_number_Of_Variables
        for convolutional_layer in range(self.__convolutions_per_layer):
            if first_convolution_boolean:
                self.__stride_tuple = (self.__upSampling_factor,)
                self.__output_padding = (self.__upSampling_factor - self.__padding,)
                first_convolution_boolean = False
            else:
                self.__stride_tuple = (1,)
                self.__output_padding = (0,)
            convolutions.append(nn.ConvTranspose1d(in_channels=int(in_channels),
                                                   out_channels=int(out_channels),
                                                   kernel_size=(int(self.__filter_size),),
                                                   stride=self.__stride_tuple,
                                                   padding=(self.__padding,),
                                                   output_padding=self.__output_padding
                                                   ))
            in_channels=out_channels
        self.__modules = list(reversed(self.__modules))
        tmp = self.__modules
        convolutions = list(reversed(convolutions))
        for i in range(self.__convolutions_per_layer):
            tmp[1 + i] = convolutions[i]
        self.__modules = tmp
        self.__modules = list(reversed(self.__modules))
        self.__moduleList = nn.ModuleList(self.__modules)

