import numpy as np
import torch
import os
import sys
import tqdm

from torch import nn

import scipy.io as sio

def load_tishby_toy_dataset(filename, assign_random_labels=False, seed=42):
    np.random.seed(seed)
    
    data = sio.loadmat(filename)
    F = data['F']
    
    if assign_random_labels:
        y = np.random.randint(0, 2)
    else:
        y = data['y'].T
    
    return F, y


class BatchGenerator():
    def __init__(self, inputs_list, batch_size, seed=None):
        self.inputs_list = inputs_list
        self.batch_size = batch_size
        self.seed = seed
    
        self.indices = np.arange(self.inputs_list[0].shape[0])
        np.random.seed(self.seed)   
        np.random.shuffle(self.indices)

    def how_it_shuffled(self):
        return [current_input[self.indices] for current_input in self.inputs_list], np.arange(len(self.indices))[self.indices]

    def batch_generator(self):
        assert(len(self.inputs_list) > 0)
        
        
        for input_array in self.inputs_list:
            assert(input_array.shape[0] == self.inputs_list[0].shape[0])
            
        data_size = self.inputs_list[0].shape[0] // self.batch_size
        
        if self.inputs_list[0].shape[0] % self.batch_size > 0:
            data_size += 1
            
        for i in range(0, data_size):
            current_indices = self.indices[i * self.batch_size: (i + 1) * self.batch_size]     
            yield [current_input[current_indices] for current_input in self.inputs_list]


class MLPWithInfo(nn.Module):
    def __init__(self, input_dim=12, layers_dim=[10, 7, 5, 4, 3, 1], 
                 activation=nn.Tanh, output_activation=nn.Sigmoid, last_activation=nn.Sigmoid):
        super().__init__()
        self.representations_per_epochs = []
        self.info_layers_numbers = []
        layers_dims = [input_dim] + layers_dim
        self.has_output_activation = output_activation is not None
        self.last_activation = last_activation

        layers = []
        
        current_layer = -1
        for i in range(len(layers_dims) - 1):
            if i != len(layers_dims) - 2: 
                layers += [nn.Linear(layers_dims[i], layers_dims[i + 1]), activation()]
                current_layer += 2

                self.info_layers_numbers.append(current_layer)
            else:
                layers += [nn.Linear(layers_dims[i], layers_dims[i + 1])]
                if output_activation is not None:
                    layers += [output_activation()]

                current_layer += 2
                self.info_layers_numbers.append(current_layer)
                 
        self.model = nn.ModuleList(layers)
        self.current_representations = None
        self.reset()
        
    def forward(self, x):
        """
        Assume that the model's layers are structured as follows:
            Linear -> activation -> Linear -> ... -> activation.
        Thus we keep every other output.
        """
        # ws_epoch = []
        current_representation = x

        # self.add_info(0, x.detach().numpy())
        next_layer_index = 0

        for i, layer in enumerate(self.model):
            current_representation = layer(current_representation)

            if i == self.info_layers_numbers[next_layer_index]:
                self.add_info(next_layer_index, current_representation.detach().numpy())
                next_layer_index += 1

        if not self.has_output_activation:
            self.add_info(next_layer_index, self.last_activation()(current_representation).detach().numpy())
        else:
            self.add_info(next_layer_index, current_representation.detach().numpy())
        # assert(len(ws_epoch) == len(self.model))
        # self.representations += ws_epoch

        return current_representation

    def next_epoch(self):
        self.representations_per_epochs.append(self.current_representations)
        self.reset()

    def add_info(self, layer_index, representations):
        if self.current_representations[layer_index] is None:
            self.current_representations[layer_index] = representations
        else:
            self.current_representations[layer_index] = np.concatenate([self.current_representations[layer_index],
                                                                       representations], axis=0)
    def reset(self):
        self.current_representations = [None for _ in range(len(self.info_layers_numbers))]
        # self.representations_epochs = []


def train_network(model, X, y, X_val, y_val, batch_size=12, epochs=16):
    """
    The network is trained with full batch
    """
    loss_list = []
    epoch_mean_loss = []
    accuracy_mean_val = []
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    loss_fun = nn.BCEWithLogitsLoss()
    model.reset()
    train_shuffles = []

    for epoch in tqdm.tqdm(range(epochs)):
        samples = 0
        cum_loss = 0

        model.reset()

        train_batcher = BatchGenerator([X, y], batch_size)
        train_shuffles.append(train_batcher.how_it_shuffled()[1])

        for X_batch, y_batch in train_batcher.batch_generator():
            X_batch = torch.Tensor(X_batch)
            y_batch = torch.Tensor(y_batch)

            model.train()
            predictions = model(X_batch)

            loss = loss_fun(predictions.reshape(-1), y_batch.reshape(-1))
            loss.backward()

            loss_list.append(loss.item())

            optimizer.step()
            optimizer.zero_grad()

            samples += X_batch.shape[0]
            cum_loss += loss.item()

        scheduler.step()
        model.next_epoch()

        epoch_mean_loss.append(cum_loss / samples)

        samples_val = 0
        accuracy_val = 0

        val_batcher = BatchGenerator([X_val, y_val], 1)

        for X_batch, y_batch in val_batcher.batch_generator():
            X_batch = torch.Tensor(X_batch)
            y_batch = torch.Tensor(y_batch)

            model.eval()
            predictions_logits = model(X_batch)

            accuracy_val += (y_batch.int() == (torch.nn.functional.sigmoid(predictions_logits) > 0.5).int()).sum().item()
            samples_val += X_batch.shape[0]

        accuracy_mean_val.append(float(accuracy_val) / samples_val)

    return epoch_mean_loss, accuracy_mean_val, train_shuffles


# def train_network_non_robus(model, X, y, X_val, y_val, epochs=16):
#     """
#     The network is trained with full batch
#     """

#     batch_size = X.shape[0]
#     loss_list = []
#     epoch_mean_loss = []
#     accuracy_mean_val = []
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     loss_fun_nonrobust = torch.nn.BCELoss()
#     model.reset()
    
#     for epoch in tqdm.tqdm(range(epochs)):
#         samples = 0
#         cum_loss = 0

#         model.reset()

#         for X_batch, y_batch in pytorch_network.batch_generator([X, y], batch_size):
#             X_batch = torch.Tensor(X_batch)
#             y_batch = torch.Tensor(y_batch)

#             model.train()
#             predictions = model(X_batch)

#             loss = loss_fun_nonrobust(predictions.reshape(-1), y_batch.reshape(-1))
#             loss.backward()

#             loss_list.append(loss.item())

#             optimizer.step()
#             optimizer.zero_grad()

#             samples += X_batch.shape[0]
#             cum_loss += loss.item()

#         model.next_epoch()

#         epoch_mean_loss.append(cum_loss / samples)

#         samples_val = 0
#         accuracy_val = 0

#         for X_batch, y_batch in pytorch_network.batch_generator([X_val, y_val], 1):
#             X_batch = torch.Tensor(X_batch)
#             y_batch = torch.Tensor(y_batch)

#             model.eval()
#             predictions_logits = model(X_batch)

#             accuracy_val += (y_batch.int() == (predictions_logits > 0.5).int()).sum().item()
#             samples_val += X_batch.shape[0]

#         accuracy_mean_val.append(float(accuracy_val) / samples_val)

#     return epoch_mean_loss, accuracy_mean_val

# nonrobust_train = train_network_non_robus(non_robust_model, X_train, y_train.astype(np.int),
#                                           X_test, y_test.astype(np.int), epochs)

# ws_nonron = non_robust_model.representations_epochs
