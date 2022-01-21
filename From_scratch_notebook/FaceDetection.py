#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
torch.cuda.empty_cache()


# In[ ]:


from runMultivariate2018DatasetChannel import runMultivariate2018DatasetChannel
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


datasetName = "FaceDetection"
normalization = True
data_location = "../Transformer/Data/Multivariate_ts"
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Hyper-parameters
filter_size_autoEncoder=3
n_layers_autoEncoder=1
convolutions_per_layer_autoEncoder=1
n_filters_index_autoEncoder=2
reducing_length_factor_autoEncoder=2
embedding_size=128
feedForward_dimension=256
nHeads=8
dropout=0.1
n_encoders=3

experiments_runner = runMultivariate2018DatasetChannel(filter_size_autoEncoder = filter_size_autoEncoder,
                                                       n_layers_autoEncoder = n_layers_autoEncoder,
                                                       convolutions_per_layer_autoEncoder = convolutions_per_layer_autoEncoder,
                                                       n_filters_index_autoEncoder = n_filters_index_autoEncoder,
                                                       reducing_length_factor_autoEncoder = reducing_length_factor_autoEncoder,
                                                       embedding_size = embedding_size,
                                                       feedForward_dimension = feedForward_dimension,
                                                       nHeads = nHeads,
                                                       dropout = dropout,
                                                       n_encoders = n_encoders,
                                                       device = device,
                                                       dataSetName = datasetName,
                                                       dataLocation = data_location,
                                                       normalization = normalization)
experiments_runner.run(n_exp=5,
                       epochs=3500,
                       batch_size=64,
                       learning_rate=1e-4,
                       ReduceLROnPlateau=False,
                       optimizer='RAdam',
                       factor=0.1,
                       min_lr=1e-18,
                       patience=10,
                       channel_sizes=[3, 5, 10],
                       file_to_write_path="results")

