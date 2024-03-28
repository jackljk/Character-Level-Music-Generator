import torch
import torch.nn as nn
from torch.autograd import Variable
from constants import *


class SongRNN(nn.Module):
    def __init__(self, input_size, output_size, config, device):
        """
        Initialize the SongRNN model.
        """
        super(SongRNN, self).__init__()

        HIDDEN_SIZE = config["hidden_size"]
        NUM_LAYERS = config["no_layers"]
        MODEL_TYPE = config["model_type"]
        DROPOUT_P = config["dropout"]

        self.model_type = MODEL_TYPE
        self.input_size = input_size
        self.hidden_size = HIDDEN_SIZE
        self.output_size = output_size
        self.num_layers = NUM_LAYERS
        self.dropout = DROPOUT_P
        self.device = device
        self.hidden = None
        """
        Complete the code

        (i) Initialize embedding layer with input_size and hidden_size
        (ii) Initialize the recurrent layer based on model type (i.e., LSTM or RNN) using hidden size and num_layers
        (iii) Initialize linear output layer using hidden size and output size
        (iv) Initialize dropout layer with dropout probability
        """
        # Initialize embedding layer w input_size and hidden_size
        self.embedding = nn.Embedding(input_size, HIDDEN_SIZE)
        
        # Initilialize the reccurent layer base don model type
        if MODEL_TYPE == "LSTM":
            self.rnn = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT_P)
        elif MODEL_TYPE == "RNN":
            self.rnn = nn.RNN(HIDDEN_SIZE, HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT_P)
        else:
            raise ValueError("Invalid model type.")
        
        # Initialize linear output layer using hidden size and output size
        self.linear = nn.Linear(HIDDEN_SIZE, output_size)
        # Initialize dropout layer with dropout probability
        self.dropout_layer = nn.Dropout(p=DROPOUT_P)
        
    def init_hidden(self): # temporarily added new argument
        """
        Initializes the hidden state for the recurrent neural network.

        Check the model type to determine the initialization method:
        (i) If model_type is LSTM, initialize both cell state and hidden state.
        (ii) If model_type is RNN, initialize the hidden state only.

        Initialise with zeros.
        """
        if self.model_type == "LSTM":
            self.hidden = (torch.zeros(self.num_layers, self.hidden_size, device=self.device),
                    torch.zeros(self.num_layers, self.hidden_size, device=self.device))
            return 
        elif self.model_type == "RNN":
            self.hidden = torch.zeros(self.num_layers, self.hidden_size, device=self.device)
            return 
        else:
            raise ValueError("Invalid model type.")
        
    def forward(self, seq):
        """
        Forward pass of the SongRNN model.
        (Hint: In teacher forcing, for each run of the model, input will be a single character
        and output will be pred-probability vector for the next character.)

        Parameters:
        - seq (Tensor): Input sequence tensor of shape (seq_length)

        Returns:
        - output (Tensor): Output tensor of shape (output_size)
        - activations (Tensor): Hidden layer activations to plot heatmap values


        TODOs:
        (i) Embed the input sequence
        (ii) Forward pass through the recurrent layer
        (iii) Apply dropout (if needed)
        (iv) Pass through the linear output layer
        """
        # Embed the input sequence
        embedded_seq = self.embedding(seq).to(self.device)
        
        # Forward pass through the recurrent layer
        
        x, self.hidden = self.rnn(embedded_seq, self.hidden)
        
        # Apply droput (if needed)
        x = self.dropout_layer(x)
        #x is the activation after the drop out
        activation = x
        # Pass through linear output layer
        x = self.linear(x)
        
        return x, activation