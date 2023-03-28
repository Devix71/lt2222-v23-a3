import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt


# defining a dataset class for storing the data from the file
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

# defining the model class for the neural network model
class Model(nn.Module):
    # adding options to the model for hidden layer, choice of two non-linearities and size of the hidden layer  
    def __init__(self, dimensions, num_hidden_layers=1, hidden_size=5, nonlinearity_func=None) -> None:
        """ is called when model is created: e.g model = Model()
            - definition of layers
        """
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.nonlinearity_func = nonlinearity_func

        self.input_layer = nn.Linear(dimensions, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nonlinearity_func() if nonlinearity_func is not None else nn.Identity()
            ) for _ in range(num_hidden_layers)
        ])
        self.output = nn.Linear(hidden_size, 3)

        # implementing the LogSoftMax for outputting the distribution of probabilities
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, data):
        """ is called when the model object is called: e.g. model(sample_input)
            - defines, how the input is processed with the previuosly defined layers 
        """
        after_input_layer = self.input_layer(data)
        for hidden_layer in self.hidden_layers:
            after_hidden_layer = hidden_layer(after_input_layer)
            after_input_layer = after_hidden_layer
        output = self.output(after_hidden_layer)

        return self.LogSoftmax(output)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("num_hidden_layers", type=int, help="Number of hidden layers in the neural network.")
    parser.add_argument("hidden_size", type=int, help="Number of neurons in each hidden layer of the neural network.")
    parser.add_argument("nonlinearity_func", type=str, help="Non-linearity function to be applied in each hidden layer. Choose between 'relu' and 'sigmoid'.")

    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()

    df_csv =  pd.read_csv(args.featurefile)

    # selecting rows with type "Train" and creating the training dataframe "df_train"
    df_train = df_csv.loc[df_csv['type'] == 'Train']
    df_train = df_train.drop(['type'], axis=1)
    column_names_x = df_train.columns[:(len(df_train.columns)-1)]

    column_names_y = df_train.columns[-1:]

    # selecting rows with type "Test" and creating the testing dataframe "df_test"
    df_test = df_csv.loc[df_csv['type'] == 'Test']
    df_test = df_test.drop(['type'], axis=1)

    y_train = df_train.iloc[:, -1:].values.flatten()

    X_train = df_train.iloc[:, :(len(df_train.columns)-1)].values

    # formating the data into a list of elements of shape tuple(feature vector,label)
    sample_list = []
    for i in range(len(y_train)):
        sample_list.append((list(X_train[i]), y_train[i]))



    # loading the data into the dataset class
    dataset = MyDataset(samples = sample_list)

    # creation of dataloader for batching and shuffling of samples
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=True,
                            collate_fn=lambda x: x)
    precision = []
    recall = []

    if args.nonlinearity_func == "Tanh":
        nonlinfunc = nn.Tanh
    elif args.nonlinearity_func == "ReLU":
        nonlinfunc = nn.ReLU
    elif args.nonlinearity_func == "None":
        nonlinfunc = None
    else:
        print("Please input a valid non-linearity function name")
        quit()

    for layer in range(1,int(args.num_hidden_layers)+1):
        model = Model(dimensions=len(df_train.columns)-1,num_hidden_layers=layer, hidden_size=args.hidden_size, nonlinearity_func=nonlinfunc)

        # optimizer defines the method how the model is trained
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        # the loss function calculates the 'difference' between the models output and the ground truth
        loss_function = nn.CrossEntropyLoss()

        # number of epochs = how often the model sees the complete dataset
        for epoch in range(5):
            total_loss = 0

            # loop through batches of the dataloader
            for i, batch in enumerate(dataloader):

                # turning the list of complete samples into a list of inputs and a list of ground_truths
                # converting both lists into a tensor (matrix), to be used by PyTorch
                model_input = torch.Tensor([sample[0] for sample in batch])
                ground_truth = torch.clamp(torch.Tensor([sample[1] for sample in batch]), min=1, max=2)  

                # sending the batch of sentences to the forward function of the model
                output = model(model_input)

                # comparing the output of the model to the ground truth and calculating the loss
                # the lower the loss, the closer the model's output is to the ground truth
                loss = loss_function(output, ground_truth.long())

                # printing average loss for the epoch
                total_loss += loss.item()
                print(f'epoch {epoch},', f'batch {i}:', round(total_loss / (i + 1), 4), end='\r')

                # training the model based on the loss:

                # computing gradients
                loss.backward()
                # updating parameters
                optimizer.step()
                # reseting gradients
                optimizer.zero_grad()
            print()
        
        # Creating the testing data
        y_test = df_test.iloc[:, -1].values.flatten()
        X_test = df_test.iloc[:, :(len(df_train.columns)-1)].values

        sample_list_test = []
        for i in range(len(y_test)):
            sample_list_test.append((list(X_test[i]), y_test[i]))

        dataset_test = MyDataset(samples = sample_list_test)

        # creating a new DataLoader for the test dataset
        test_dataloader = DataLoader(dataset_test,
                                    batch_size=4,
                                    shuffle=False,
                                    collate_fn=lambda x: x)
        
        # predicting labels for the test set
        model.eval()
        with torch.no_grad():
            model_input = torch.Tensor(X_test)
            output = model(model_input)
            y_pred = torch.argmax(output, dim=1).numpy()

        

        # appending the precision and recall scores to the arrays
        precision.append(precision_score(y_test,y_pred, zero_division = 1) )
        recall.append( recall_score(y_test,y_pred, zero_division = 1))


    # plot precision vs. recall for each layer
    for i in range(len(precision)):
        plt.scatter(recall[i], precision[i], label=f'P vs. R of Layer {i+1}')

    # set axis labels and plot title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    # show the legend and plot
    plt.legend()
    plt.savefig('precision_recall_curve.png')

