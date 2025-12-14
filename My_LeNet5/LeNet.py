import torch
from torch import nn   # layer, stimulation function, CNN..., pool
from torchsummary import summary

class LeNet(nn.Module):

    model:str = "LeNet5"


    def __init__(self):
        """
        input 28x28x1

        sigmoid()
        conv1 -> 28x28x6
        pool1 -> 14x14x6
        conv2 -> 10x10x16
        sigmoid()
        pool2 -> 5x5x16
        flatten -> 400
        linear1 -> 120
        sigmoid()
        linear2 -> 84
        sigmoid()
        linear3 -> 10

        output
        """
        
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1)
        self.stimulation_function = nn.Sigmoid()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=400, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.linear3 = nn.Linear(in_features=84, out_features=10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        parameters:
        x: input data -> Torch Tensor
        """
        x = self.stimulation_function(self.conv1(x))
        x = self.pool1(x)
        x = self.stimulation_function(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.stimulation_function(self.linear1(x))
        x = self.stimulation_function(self.linear2(x))
        x = self.linear3(x)
        return x

    def __str__(self):
        return(self.model)


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("___CURRENT DEVICE___: ", device)

    # Load model
    model = LeNet().to(device)
    model_summary = summary(model, (1, 28, 28))
    print(model_summary)





# upgrade tunnel  decline spatial resolution

# Code: model train (data processing, loss..., weight & bias)
# Train Verification Test

# ->