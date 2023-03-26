import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, samples) -> None:
        """ is called, when creating the object: e.g. dataset = MyDataset(samples)
            - stores samples in instance variable self.samples 
        """
        super().__init__()
        self.samples = samples

    def __getitem__(self, idx):
        """ is called when object is accesed with squre brackets: e.g. dataset[3]
        """
        return self.samples[idx]

    def __len__(self):
        """ is called by the magic function len: e.g. len(dataset)
        """
        return len(self.samples)


class Model(nn.Module):
    def __init__(self) -> None:
        """ is called when model is created: e.g model = Model()
            - definition of layers
        """
        super().__init__()

        self.input_layer = nn.Linear(3, 5)
        self.hidden = nn.Linear(5,5)
        self.output = nn.Linear(5,4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        """ is called when the model object is called: e.g. model(sample_input)
            - defines, how the input is processed with the previuosly defined layers 
        """
        after_input_layer = self.input_layer(data)
        after_hidden_layer = self.hidden(after_input_layer)
        output = self.output(after_hidden_layer)
        
        return self.softmax(output)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()

    df_csv =  pd.read_csv(args.featurefile)


    # Select rows with type "Train" and create a new dataframe "df_train"
    df_train = df_csv.loc[df_csv['type'] == 'Train']
    df_train = df_train.drop(['type'], axis=1)
    column_names_x = df_train.columns[:3]

    column_names_y = df_train.columns[-1:]

    # Select rows with type "Test" and create a new dataframe "df_test"
    df_test = df_csv.loc[df_csv['type'] == 'Test']
    df_test = df_test.drop(['type'], axis=1)

    y_train = df_train.iloc[:, -1:].values.flatten()
    X_train = df_train.iloc[:, :3].values

    sample_list = []
    for i in range(len(y_train)):
        sample_list.append((list(X_train[i]), y_train[i]))




    dataset = MyDataset(samples = sample_list)
    # creation of dataloader for batching and shuffling of samples
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=True,
                            collate_fn=lambda x: x)
    model = Model()

    # optimizer defines the method how the model is trained
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # the loss function calculates the 'difference' between the models output and the ground thruth
    loss_function = nn.CrossEntropyLoss()

    # number of epochs = how often the model sees the complete dataset
    for epoch in range(4):
        total_loss = 0
        # loop through batches of the dataloader
        for i, batch in enumerate(dataloader):
            # turn list of complete samples into list of inputs and list of ground_truths
            # both lists are then converted into a tensor (matrix), which can be used by PyTorch
            model_input = torch.Tensor([sample[0] for sample in batch])
            ground_truth = torch.Tensor([sample[1] for sample in batch])

            # send your batch of sentences to the forward function of the model
            output = model(model_input)

            # compare the output of the model to the ground truth to calculate the loss
            # the lower the loss, the closer the model's output is to the ground truth
            loss = loss_function(output, ground_truth.long())


            # print average loss for the epoch
            total_loss += loss.item()
            print(f'epoch {epoch},', f'batch {i}:', round(total_loss / (i + 1), 4), end='\r')

            # train the model based on the loss:
            # compute gradients
            loss.backward()
            # update parameters
            optimizer.step()
            # reset gradients
            optimizer.zero_grad()
        print()
    
    # Create the training data

    y_test = df_test.iloc[:, -1].values
    X_test = df_test.iloc[:, :3].values


    sample_list_test = []
    for i in range(len(y_test)):
        sample_list_test.append((list(X_test[i]), y_test[i]))

    #print(sample_list_test)

    dataset_test = MyDataset(samples = sample_list_test)

    # create a new DataLoader for the test dataset
    test_dataloader = DataLoader(dataset_test,
                                batch_size=4,
                                shuffle=False,
                                collate_fn=lambda x: x)
    # Get predicted labels for the test set
    model.eval()
    with torch.no_grad():
        model_input = torch.Tensor(X_test)
        output = model(model_input)
        y_pred = torch.argmax(output, dim=1).numpy()

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)


        # plot the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(np.arange(len(column_names_x)), column_names_x, rotation=45)
    plt.yticks(np.arange(len(column_names_y)), column_names_y)
    plt.tight_layout()
    plt.show()

    # Print confusion matrix
    print(cm)
    '''    # switch model to evaluation mode
        model.eval()

        # make predictions on the test data
        predictions = []
        with torch.no_grad():
            for sample in dataset_test:
                model_input = torch.Tensor(sample[0])
                output = model(model_input.unsqueeze(0))
                _, predicted = torch.max(output.data, 1)
                predictions.append(predicted.item())

        # compare the predictions to the ground truth labels
        y_true = y_test.tolist()
        accuracy = sum([1 for i in range(len(y_true)) if y_true[i] == predictions[i]]) / len(y_true)
        print(f'Test accuracy: {accuracy:.4f}')
    '''





    ''' # Make predictions on the test data
    X_test = torch.Tensor([sample_list_test])
    y_pred = model(X_test)
    predictions = torch.argmax(y_pred, dim=1).numpy()

    print(predictions)
    print(y_test)'''
