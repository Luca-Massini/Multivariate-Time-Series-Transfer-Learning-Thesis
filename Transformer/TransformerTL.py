import copy
from torch.utils.data import DataLoader as Loader
from Transformer.ConvolutionalDecoder import ConvolutionalDecoder
from Transformer.ConvolutionalEncoder import ConvolutionalEncoder
from Transformer.TransformerTL_MultiLoss import TransformerTL_Loss
from Transformer.Transformer_MVTS import *
import importlib
import numpy as np
import tqdm as tq
from Transformer.optimizers import RAdam
import matplotlib.pyplot as plt


class TransformerTL(nn.Module):

    def __init__(self,
                 filter_size_autoEncoder: int,
                 n_layers_autoEncoder: int,
                 convolutions_per_layer_autoEncoder: int,
                 n_filters_index_autoEncoder: int,
                 autoEncoder_encoding_channel_size: int,
                 reducing_length_factor_autoEncoder: int,
                 embedding_size: int,
                 time_series_channels_number: int,
                 feedForward_dimension: int,
                 nHeads: int,
                 dropout: float,
                 time_series_length: int,
                 n_classes: int,
                 n_encoders: int,
                 device: torch.device):
        super().__init__()

        self.__embedding_size = embedding_size
        self.__downSampling_factor = reducing_length_factor_autoEncoder
        self.__n_layers_autoEncoder = n_layers_autoEncoder
        self.__dropout = dropout

        self.__length_after_autoEncoder = self.__compute_autoEncoder_encoding_length(length=time_series_length,
                                                                                     down_sampling_factor=reducing_length_factor_autoEncoder,
                                                                                     n_layers=n_layers_autoEncoder)

        self.__transformer = TSTransformerEncoderClassiregressor(feat_dim=autoEncoder_encoding_channel_size,
                                                                 max_len=self.__length_after_autoEncoder,
                                                                 d_model=embedding_size, n_heads=nHeads,
                                                                 num_layers=n_encoders,
                                                                 dim_feedforward=feedForward_dimension,
                                                                 num_classes=n_classes,
                                                                 dropout=dropout,
                                                                 pos_encoding='learnable',
                                                                 activation='relu',
                                                                 norm='BatchNorm',
                                                                 freeze=False)

        self.__DEVICE = device

        self.__convolutional_encoder = ConvolutionalEncoder(channel_length=time_series_channels_number,
                                                            filter_size=filter_size_autoEncoder,
                                                            n_layers=n_layers_autoEncoder,
                                                            convolutions_per_layer=convolutions_per_layer_autoEncoder,
                                                            n_filters_index=n_filters_index_autoEncoder,
                                                            final_number_filters=autoEncoder_encoding_channel_size,
                                                            down_sampling_factor=reducing_length_factor_autoEncoder,
                                                            dropout=dropout)

        self.__convolutional_decoder = ConvolutionalDecoder(input_channel_size=autoEncoder_encoding_channel_size,
                                                            filter_size=filter_size_autoEncoder,
                                                            n_layers=n_layers_autoEncoder,
                                                            convolutions_per_layer=convolutions_per_layer_autoEncoder,
                                                            n_filters_index=n_filters_index_autoEncoder,
                                                            before_encoding_channel_size=self.__convolutional_encoder.get_last_channel_size_before_encoding(),
                                                            original_channel_length=time_series_channels_number,
                                                            upSampling_factor=reducing_length_factor_autoEncoder,
                                                            dropout=dropout)

    def forward(self, x):
        x, padding = self.__convolutional_encoder(x)
        time_embeddings = torch.transpose(x, -1, -2)
        batch_size = x.shape[0]
        seq_length = self.__length_after_autoEncoder
        padding_mask = torch.tensor(np.ones((batch_size, seq_length)) > 0, requires_grad=False)
        classification_scores = self.__transformer(X=time_embeddings,
                                                   padding_masks=padding_mask.to(self.__DEVICE))
        reconstructed_time_series = self.__convolutional_decoder(x, padding)
        return classification_scores, reconstructed_time_series

    def getAutoEncoder(self):
        return self.__convolutional_encoder, self.__convolutional_decoder

    @staticmethod
    def __compute_autoEncoder_encoding_length(length, down_sampling_factor, n_layers):
        reducing_factor = down_sampling_factor ** n_layers
        length_after_encoding = length // reducing_factor
        if length % reducing_factor != 0:
            length_after_encoding += 1
        return length_after_encoding

    def get_transfer_learning_model(self, new_ts_length, new_number_of_variables, new_n_classes):
        result = copy.deepcopy(self)
        result.__convolutional_encoder.adapt_to_new_Dataset(new_number_Of_Variables=new_number_of_variables)
        result.__convolutional_decoder.adapt_to_new_Dataset(new_number_Of_Variables=new_number_of_variables)
        result.__length_after_autoEncoder = result.__compute_autoEncoder_encoding_length(length=new_ts_length,
                                                                                         down_sampling_factor=result.__downSampling_factor,
                                                                                         n_layers=result.__n_layers_autoEncoder)
        result.__transformer.output_layer = result.__transformer.build_output_module(d_model=result.__embedding_size,
                                                                                     max_len=result.__length_after_autoEncoder,
                                                                                     num_classes=new_n_classes)
        result.__transformer.pos_enc = get_pos_encoder('learnable')(result.__embedding_size,
                                                                    dropout=result.__dropout * (1.0 - False),
                                                                    max_len=result.__length_after_autoEncoder)
        return result

    def freeze_model(self):
        for name, param in self.named_modules():
            if name.endswith("output_layer"):
                param.requires_grad = True
            else:
                if name.endswith("_ConvolutionalEncoder__moduleList.0"):
                    param.requires_grad = True
                else:
                    if name.endswith(
                            "ConvolutionalDecoder__moduleList." + str((3 * self.__n_layers_autoEncoder) - 3)):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        return self

    def fit(self,
            trainingSet,
            batchSize: int,
            epochs: int,
            learning_rate: float,
            ReduceLROnPlateau: bool,
            optimizer: str,
            factor=0.1,
            min_lr=1e-18,
            patience=25,
            validationDataset=None,
            print_accuracy_test=False,
            verbose=False,
            weight_decay=0,
            training_loss_functions_plot=False,
            freeze=False,
            compute_accuracy_every: int = 1):

        learning_rate_scheduler = None

        if freeze:
            model = self.freeze_model()
        else:
            model = self

        training_accuracy_values = []
        validation_accuracy_values = []

        classification_loss_values = []
        time_series_reconstruction_loss_values = []
        multitask_learning_loss_value = []

        model = model.to(self.__DEVICE)

        # Dataloader used for the training
        trainLoader = Loader(trainingSet, batch_size=batchSize,
                             shuffle=True, num_workers=2)

        # import the right optimizer
        optimizer_module = importlib.import_module("torch.optim")
        # if the provided optimizer name is None, the default choice is Adam
        if optimizer != 'RAdam':
            if optimizer is None:
                optimizer = 'Adam'
            optimizer_class = getattr(optimizer_module, optimizer)
            if weight_decay != 0:
                optimizer = optimizer_class(filter(lambda p: p.requires_grad, model.parameters()),
                                            lr=learning_rate,
                                            weight_decay=weight_decay)
            else:
                optimizer = optimizer_class(filter(lambda p: p.requires_grad, model.parameters()),
                                            lr=learning_rate)
        else:
            optimizer = RAdam(lr=learning_rate,
                              params=filter(lambda p: p.requires_grad, model.parameters()),
                              weight_decay=weight_decay)

        # loss function to be minimized
        criterion = TransformerTL_Loss(verbose=verbose)

        # if the learning rate scheduler name is provided, it is imported and used
        if ReduceLROnPlateau:
            scheduler_module = importlib.import_module("torch.optim.lr_scheduler")
            scheduler_class = getattr(scheduler_module, "ReduceLROnPlateau")
            learning_rate_scheduler = scheduler_class(optimizer,
                                                      mode='min',
                                                      factor=factor,
                                                      min_lr=min_lr,
                                                      verbose=False,
                                                      patience=patience)

        # The best training loss value reached so far. Since the training is not started yet is is +inf
        average_classification_loss_value = 0
        average_loss = 0
        average_encoding_loss = 0

        index = 0
        for _ in tq.tqdm(range(epochs)):
            # loss = 0
            running_classification_loss = 0.0
            running_encoding_loss = 0.0
            running_loss = 0.0
            for i, data in enumerate(trainLoader, 0):
                x, y, _ = data
                inputs, labels = x.float().to(self.__DEVICE), y.float().to(self.__DEVICE)

                optimizer.zero_grad()
                class_prediction, reconstructed_time_series = model(x=inputs.float().to(self.__DEVICE))
                loss = criterion(true_class=torch.max(labels, 1)[1].to(self.__DEVICE),
                                 predicted_class=class_prediction.to(self.__DEVICE),
                                 reconstructed_time_series=reconstructed_time_series.to(self.__DEVICE),
                                 original_time_series=x.float().to(self.__DEVICE))
                loss.backward()
                optimizer.step()

                running_classification_loss += criterion.get_last_classification_loss_value()
                mini_batch_counter = i + 1
                average_classification_loss_value = running_classification_loss / mini_batch_counter
                if training_loss_functions_plot:
                    running_encoding_loss += criterion.get_last_reconstruction_loss_value()
                    average_encoding_loss = running_encoding_loss / mini_batch_counter
                    running_loss += criterion.get_last_loss_value()
                    average_loss = running_loss / mini_batch_counter
            if training_loss_functions_plot:
                classification_loss_values.append(average_classification_loss_value)
                multitask_learning_loss_value.append(average_loss)
                time_series_reconstruction_loss_values.append(average_encoding_loss)

            # if the provided learning rate_scheduler name is not None, the right learning rate scheduler is used during
            # the training procedure otherwise no learning rate scheduler is used
            if ReduceLROnPlateau:
                learning_rate_scheduler.step(average_classification_loss_value)

            if index + 1 == compute_accuracy_every:
                if print_accuracy_test or validationDataset is not None:
                    if validationDataset is not None:
                        validation_accuracy_values.append(self.printAccuracy(dataSet=validationDataset,
                                                                             test_model=model,
                                                                             DEVICE=self.__DEVICE,
                                                                             batch=batchSize,
                                                                             train_validation="validation",
                                                                             print_=print_accuracy_test))
                    training_accuracy_values.append(self.printAccuracy(dataSet=trainingSet,
                                                                       test_model=model,
                                                                       DEVICE=self.__DEVICE,
                                                                       batch=batchSize,
                                                                       train_validation="training",
                                                                       print_=print_accuracy_test))
                index = 0
            else:
                index += 1
        n_accuracy_computations = int(epochs/compute_accuracy_every)
        epochs = [i*compute_accuracy_every for i in range(n_accuracy_computations)]
        if training_loss_functions_plot:
            plt.plot(epochs, multitask_learning_loss_value)
            plt.plot(epochs, classification_loss_values)
            plt.plot(epochs, time_series_reconstruction_loss_values)
            plt.legend(['multitask learning loss', 'CrossEntropy loss (classification loss)',
                        'time series reconstruction loss (MSE)'], loc='upper left')
            plt.show()

        if validationDataset is not None:
            best_validation_accuracy = np.max(validation_accuracy_values)
            related_training_accuracy = training_accuracy_values[np.argmax(validation_accuracy_values)]
            line = "\nbest validation accuracy: " + str(best_validation_accuracy) + " related to a training accuracy of: " + str(related_training_accuracy)
            return best_validation_accuracy, line
        else:
            return None, None

    @staticmethod
    def printAccuracy(dataSet, test_model, DEVICE, batch=8, train_validation='training', print_=False):
        testLoader = torch.utils.data.DataLoader(dataSet,
                                                 batch_size=batch,
                                                 shuffle=False,
                                                 num_workers=2)
        correct = 0
        total = 0

        test_model.eval()

        with torch.no_grad():
            for data in testLoader:
                time_series, labels, _ = data
                time_series, labels = time_series.float().to(DEVICE), labels.float().to(DEVICE)
                outputs, _ = test_model(x=(time_series.float()).to(DEVICE))

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == torch.max(labels, 1)[1]).sum().item()

        accuracy = 100 * correct / total
        test_model.train()
        if print_:
            print("\n accuracy on the", train_validation, "set is: ", accuracy)
        return accuracy
