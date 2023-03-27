
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
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
    def __init__(self, num_hidden_layers=1, hidden_size=5, nonlinearity_func=nn.ReLU) -> None:
        """ is called when model is created: e.g model = Model()
            - definition of layers
        """
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.nonlinearity_func = nonlinearity_func

        self.input_layer = nn.Linear(3, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nonlinearity_func()
            ) for _ in range(num_hidden_layers)
        ])
        self.output = nn.Linear(hidden_size, 3)
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

        return self.LogSoftmax(output), output

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


    # Select rows with type "Train" and create a new dataframe "df_train"
    df_train = df_csv.loc[df_csv['type'] == 'Train']
    df_train = df_train.drop(['type'], axis=1)
    column_names_x = df_train.columns[:3]

    column_names_y = df_train.columns[-3:]

    # Select rows with type "Test" and create a new dataframe "df_test"
    df_test = df_csv.loc[df_csv['type'] == 'Test']
    df_test = df_test.drop(['type'], axis=1)

    y_train = df_train.iloc[:, -3:].values
    X_train = df_train.iloc[:, :3].values

    sample_list = []

    for i in range(len(y_train)):
        sample_list.append((list(X_train[i]),list(y_train[i]) ))



    
    dataset = MyDataset(samples = sample_list)
    # creation of dataloader for batching and shuffling of samples
    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=True,
                            collate_fn=lambda x: x)
    

    model = Model(num_hidden_layers=2, hidden_size=10, nonlinearity_func=nn.Tanh)


    # optimizer defines the method how the model is trained
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # the loss function calculates the 'difference' between the models output and the ground truth
    loss_function = nn.CrossEntropyLoss()

    # number of epochs = how often the model sees the complete dataset
    for epoch in range(4):
        total_loss = 0
        # loop through batches of the dataloader
        for i, batch in enumerate(dataloader):
            # turn list of complete samples into list of inputs and list of ground_truths
            # both lists are then converted into a tensor (matrix), which can be used by PyTorch
            model_input = torch.Tensor([sample[0] for sample in batch])
            ground_truth = torch.Tensor([sample[1] for sample in batch]).long()

            # send your batch of sentences to the forward function of the model
            log_probs, output = model(model_input)

            # compare the output of the model to the ground truth to calculate the loss
            # the lower the loss, the closer the model's output is to the ground truth
            loss = loss_function(output, ground_truth.float())

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
    
    # Create the training data

    y_test = df_test.iloc[:, -3].values
    X_test = df_test.iloc[:, :3].values


    sample_list_test = []
    for i in range(len(y_test)):
        sample_list_test.append((list(X_train[i]),list(y_train[i]) ))

    #print(sample_list_test)

    dataset_test = MyDataset(samples = sample_list_test)

    # create a new DataLoader for the test dataset
    test_dataloader = DataLoader(dataset_test,
                                batch_size=1,
                                shuffle=False,
                                collate_fn=lambda x: x)
    model.eval()
    with torch.no_grad():
        model_input = torch.Tensor(X_test)
        output = model(model_input)
        
        y_pred = torch.argmax(output[0], dim=1).numpy()
        probabilities = F.softmax(output[0], dim=1)
        class_probabilities = probabilities.numpy()
        print(class_probabilities)

    # Compute confusion matrix

    # Get the predicted labels
    y_pred = np.argmax(class_probabilities, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])


    print(cm)
    # Plot confusion matrix
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, column_names_y, rotation=45)
    plt.yticks(tick_marks, column_names_y)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # Display counts in confusion matrix
    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.show()
