import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for classifying text data into fake or real news categories.

    This network consists of three hidden layers and one output layer.

    Attributes:
        fc1, fc2, fc3, fc4 (nn.Linear): Fully connected layers.
        silu1, silu2, silu3, silu4 (nn.SiLU): Activation functions for each hidden layer.
        layernorm (nn.LayerNorm): Layer normalization for hidden layer 1.
        dropout, dropout2 (nn.Dropout): Dropout layers for regularization.
        sigmoid (nn.Sigmoid): Activation function for the output layer.
    """

    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        Initializes the MLP model with specified dimensions.

        Args:
            input_dim (int): The dimension of the input layer.
            hidden_dims (list[int]): A list of dimensions for the hidden layers.
            output_dim (int): The dimension of the output layer.
        """
        super(MLP, self).__init__()
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip([input_dim] + hidden_dims[:-1], hidden_dims)):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.SiLU())
            if i == 0:
                layers.append(nn.LayerNorm(out_dim))
            if i in [1, 2]:  # Apply dropout after the 2nd and 3rd hidden layers
                layers.append(nn.Dropout(0.2 if i == 1 else 0.3))

        self.layers = nn.Sequential(*layers)
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Defines the forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the MLP.
        """
        x = self.layers(x)
        return self.output(x)