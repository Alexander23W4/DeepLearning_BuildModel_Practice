from time import time
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from LeNet import LeNet
import copy
import pandas as pd
import time

def Input_Data_Process():
    """
    Return batch of data for train or test
    
    """
    # Load and preprocessing data to the program
    data = FashionMNIST(
        root='./data',
        train=True,
        transform=transforms.ToTensor(), # tranfrom PIL image to Torch Tensor
        download=True)
    
    # Split train & test data from all input data with ratio 8:2
    train_data, test_data = Data.random_split(data, [round(len(data)*0.8), round(len(data)*0.2)])

    # Load train & test data to DataLoader (batch size and other para setting bases on specific circumstance)
    """
    DataLoader parameter notes:
    - dataset: A dataset object implementing __len__ and __getitem__ (here we use the subsets returned by random_split).
    - batch_size: Number of samples per batch. Larger batch sizes can improve GPU utilization but use more memory; choose based on available memory and convergence behavior (common values: 32, 64).
    - shuffle: Whether to shuffle the data at the start of each epoch. Set True for training to randomize input order, and False for testing/validation to keep evaluation deterministic.
    - num_workers: Number of subprocesses used for data loading. Using (larger than 0) enables parallel loading and can increase throughput, but too many workers can add overhead or cause issues on Windows a typical choice is the number of CPU cores or a smaller value.

    """

    train_loader = Data.DataLoader(dataset=train_data, 
                                   batch_size=32,
                                   shuffle=True,
                                   num_workers=2)
    
    test_loader = Data.DataLoader(dataset=test_data,
                                  batch_size=32,
                                  shuffle=False,
                                  num_workers=2)
    
    return train_loader, test_loader

def Train_Model_Process(model, train_loader, test_loader, EPOCH: int):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("___CURRENT DEVICE___: ", device)

    # Move the provided model to device (don't re-instantiate)
    model = model.to(device)

    # Loss function: CrossEntropyLoss
    loss_function = torch.nn.CrossEntropyLoss()

    # Use Adam optimizer, learning rate 0.001 - use the model instance parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Keep best model weights
    best_model_wts = copy.deepcopy(model.state_dict())

    # loss & accuracy record (list)
    train_loss_all = []
    test_loss_all = []
    train_accuracy = []
    test_accuracy = []

    best_accruacy = 0.0
    training_time = time.time()

    # Epoch: batch
    for epoch in range(EPOCH):
        print(f"___EPOCH [{epoch+1}/{EPOCH}]___")
        print("-"*20)

        # parameter (loss, correct) initialization
        train_loss = 0.0
        test_loss = 0.0
        train_correct = 0
        test_correct = 0
        train_num = 0
        test_num = 0

        # Load data to device, batch by batch (32*28*28*1)
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            
            # Train mode
            model.train()

            output = model(b_x)
            pre_label = torch.argmax(output, dim=1)   # get the index of the max log-probability
            loss = loss_function(output, b_y)

            optimizer.zero_grad()   # initialize gradient to 0
            loss.backward()         # back propagation, compute gradient
            optimizer.step()        # update weights & biases

            train_loss += loss.item() * b_x.size(0)  # loss.item() is the average loss of the batch
            train_correct += torch.sum(pre_label == b_y)
            train_num += b_x.size(0)
        
        # Validation mode
        for step, (b_x, b_y) in enumerate(test_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()

            output = model(b_x)
            pre_label = torch.argmax(output, dim=1)
            loss = loss_function(output, b_y)
            test_loss += loss.item() * b_x.size(0)
            test_correct += torch.sum(pre_label == b_y)
            test_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)  # append average loss of this epoch
        test_loss_all.append(test_loss / test_num)
        train_accuracy.append(train_correct.double().item() / train_num) #@ double accruacy (0.00)
        test_accuracy.append(test_correct.double().item() / test_num)

        # Output the model train/test loss & accuracy of each epoch
        #@ Array[-1] means the last element of the array
        print("Epoch{}: Train Loss: {:.4f}, Train Accuracy: {:.4f}%"
              .format(epoch+1, train_loss_all[-1], train_accuracy[-1]*100))
        print("Epoch{}: Test Loss: {:.4f}, Test Accuracy: {:.4f}%"
              .format(epoch+1, test_loss_all[-1], test_accuracy[-1]*100))
    
        # Deep copy the best model (store the trained model weights)
        if test_accuracy[-1] > best_accruacy:
            best_accruacy = test_accuracy[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        # Ouput time cost of each epoch
        print("Time cost: {:.0f}m {:.0f}s"
              .format((time.time() - training_time)//60, (time.time() - training_time)%60))

        # (end of epoch) continue to next epoch; don't save or return here
        
    # After training loop: load & save best model weights, and prepare DataFrame
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "./LeNet_Model.pth")

    train_process_note = pd.DataFrame(data={
        "epoch": list(range(1, len(train_loss_all) + 1)),
        "train_loss": train_loss_all,
        "test_loss": test_loss_all,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    })
    return train_process_note
    

def Plot_Train_Process(train_process_note: pd.DataFrame):
    """
    Plot the train process (loss & accuracy)
    parameters:
    train_process_note: DataFrame record the train process
    """
    # Plot loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process_note['epoch'], train_process_note['train_loss'], label='Train Loss')
    plt.plot(train_process_note['epoch'], train_process_note['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss over Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_process_note['epoch'], train_process_note['train_accuracy'], label='Train Accuracy')
    plt.plot(train_process_note['epoch'], train_process_note['test_accuracy'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    LeNet_model = LeNet()

    train_data, test_data = Input_Data_Process()
    train_process_note = Train_Model_Process(LeNet_model, train_data, test_data, EPOCH=5) 
    Plot_Train_Process(train_process_note)