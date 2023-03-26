import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from sklearn.metrics import confusion_matrix
# Whatever other imports you need

# You can implement classes and helper functions here too.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()




    print("Reading {}...".format(args.featurefile))

    # implement everything you need here
    df_csv =  pd.read_csv(args.featurefile)


    # Define the model
    class Perceptron(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(Perceptron, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            
        def forward(self, x):
            return nn.functional.log_softmax(self.linear(x), dim=1)


    



    # Select rows with type "Train" and create a new dataframe "df_train"
    df_train = df_csv.loc[df_csv['type'] == 'Train']
    df_train = df_train.drop(['type'], axis=1)

    # Select rows with type "Test" and create a new dataframe "df_test"
    df_test = df_csv.loc[df_csv['type'] == 'Test']
    df_test = df_test.drop(['type'], axis=1)

    y_train = df_train.iloc[:, -1:].values.flatten()
    X_train = df_train.iloc[:, :3].values

    sample_list = []
    for i in range(len(y_train)):
        sample_list.append((list(X_train[i]), y_train[i]))

    print(sample_list)

    # Get list of tuples
    

    ''' # Create the training data

    y_train = df_train.iloc[:, -3:].values
    X_train = df_train.iloc[:, :3].values
    

    #print(type(y_train))
    #print(X_train)
    # Convert the data to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()

    # Define the model and the optimizer
    
    input_dim = X_train.shape[1]
    output_dim = (len(np.unique(y_train))+1)
    model = Perceptron(input_dim, output_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.1)


    # Train the model
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        y_train_shaped = y_train.reshape(-1)
        y_pred_shaped = y_pred.reshape(-1)
        loss = nn.functional.nll_loss(y_pred_shaped, y_train_shaped)
        loss.backward()
        optimizer.step()


    # Create the training data

    y_test = df_test.iloc[:, -3:].values
    X_test = df_test.iloc[:, :3].values

    # Make predictions on the test data
    X_test = torch.from_numpy(X_test).float() 
    y_pred = model(X_test)
    predictions = torch.argmax(y_pred, dim=1).numpy()

    print(predictions)
    print(y_test)'''

    
