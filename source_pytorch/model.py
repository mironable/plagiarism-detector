# torch imports
import torch.nn.functional as F
import torch.nn as nn


class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    # define the init function and the required input params (for loading code in train.py to work)
    def __init__(self, input_features, hidden_dim, output_dim):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super(BinaryClassifier, self).__init__()

        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # define initial layers
        self.layer1 = nn.Linear(self.input_features, self.hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(self.hidden_dim, 100)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(100, self.output_dim)
        # sigmoid layer
        self.sig = nn.Sigmoid()

    
    ## define the feedforward behaviour of the network
    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        
        out = F.relu(self.layer1(x)) 
        out = self.relu(out)
        out = self.drop(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.out(out)
        return self.sig(out)
    
