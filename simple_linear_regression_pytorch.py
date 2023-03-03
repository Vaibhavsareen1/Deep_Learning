# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 07:30:55 2023

@author: vaibh
"""

import torch
import numpy as np
import random

class LinearRegressionModel(torch.nn.Module):
    """Class to implement linear regression on a single variable"""
    
    def __init__(self):
        super().__init__()
        
        self.linear_layer = torch.nn.Linear(in_features=1,
                                            out_features=1,
                                            bias=True,
                                            dtype=torch.float32)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Method to compute a forward pass in a neural network"""
        return self.linear_layer.forward(x)

    

if __name__ == '__main__':
    # Setting up code to be reproducible
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # setting up device for device agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create a synthetic dataset of one variable
    start = 0
    end = 1
    step = 0.02
    # Actual weight and bias
    actual_weight = 0.7
    actual_bias = 0.3
    X = torch.arange(start, end, step, requires_grad=False, dtype=torch.float32)
    X = X.unsqueeze(dim=1)
    # Load tensors on the device
    X = X.to(device)
    y = actual_weight * X + actual_bias

    # Splitting data into train and test set
    train_index = int( 0.8 * X.shape[0])
    X_train, X_test = X[: train_index], X[train_index: ]
    y_train, y_test = y[: train_index], y[train_index: ]

    model = LinearRegressionModel()
    # shifting the model to the device
    model.to(device)

    # instantiating the loss function and the optimizer
    learning_rate = 0.01
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # Setting up the training loop
    epochs = 200
    for epoch in range(epochs):
        model.train()
        
        training_y_pred = model.forward(X_train)
        loss = loss_function.forward(training_y_pred, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.inference_mode():
                train_pred = model.forward(X_train)
                test_pred = model.forward(X_test)
                
                train_loss = loss_function.forward(train_pred, y_train).item()
                test_loss = loss_function.forward(test_pred, y_test).item()
                
                print(f'epochs: {epoch} | train_loss: {train_loss} | test_loss: {test_loss}')

    # Print out the model parameters
    print(list(model.parameters()))
    # Print out the device the model parameters are on
    print(next(model.parameters()).device)
    # Print out the state of the model parameters
    print(model.state_dict())
