import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self) -> None:
        """ is called, when creating the object: e.g. dataset = MyDataset()
            - stores samples in instance variable self.samples 
        """
        super().__init__()
        self.samples = [([1, 4, 5], 3),
                        ([1, 4, 5], 1),
                        ([1, 4, 5], 2),
                        ([1, 4, 5], 1),
                        ([1, 4, 5], 2),
                        ([1, 4, 5], 1),
                        ([1, 4, 5], 2),
                        ([1, 4, 5], 1),
                        ([1, 4, 5], 2),
                        ([1, 4, 5], 1),
                        ([1, 4, 5], 2),
                        ([1, 4, 5], 1),
                        ([1, 4, 5], 2),
                        ([1, 4, 5], 1),
                        ([1, 4, 5], 2),
                        ([1, 4, 5], 1),
                        ([1, 4, 5], 2),
                        ([1, 4, 5], 1),
                        ([1, 4, 5], 2),
                        ([1, 4, 5], 1),
                        ([1, 4, 5], 2),]

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
    dataset = MyDataset()
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
