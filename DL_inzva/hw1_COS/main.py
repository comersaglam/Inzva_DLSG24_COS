import torch
import torch.nn as nn
from utils.model import NeuralNet
from utils.data_utils import *
from utils.viz_utils import *
from training import train_model
from testing import test_model

def main():
    '''
    The main function that coordinates the training and testing of the neural network on the CIFAR-10 dataset.

    Steps:
    - Sets the input size and number of classes for the CIFAR-10 dataset.
    - Defines the training parameters such as epochs, batch size, and learning rate.
    - Initializes the model, loss function, and optimizer.
    - Loads the training, validation, and test datasets.
    - Trains the model and visualizes the training and validation losses.
    - Tests the trained model on the test dataset and prints the accuracy.

    '''

    input_size = 3*32*32  # CIFAR-10 images are 3-channel RGB images with 32x32 pixels (3*32*32)
    num_classes = 10  # CIFAR-10 has 10 classes
    epochs = 50
    batch_size = 64
    learning_rate = 0.001

    # Define the device for computation (GPU if available, otherwise CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Initialize the model, loss function, and optimizer
    model = NeuralNet(input_size, num_classes).to(device)

    # Define the loss function (CrossEntropy for classification)
    criterion = nn.CrossEntropyLoss()

    # Choose optimizer with a learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


    # Get the training and test datasets
    train_set, test_set = get_train_and_test_set()
    print(len(train_set), len(test_set))
    
    # Create DataLoader for the training set
    train_loader = get_trainloader(train_set, batch_size)
    
    # Split the test set into validation and test subsets
    val_indices, test_indices = split_testset(test_set, test_size=0.5)
    
    # Create DataLoaders for validation and test subsets
    val_loader  = get_validationloader(test_set, val_indices, batch_size)
    test_loader = get_testloader(test_set, test_indices, batch_size)
    
    # Train the model and track losses for each epoch
    train_losses, val_losses = train_model(model, device, epochs, criterion, optimizer, train_loader, val_loader)
    print('Finished Training')
    
    # Visualize the training and validation loss curves
    #visualize_train_val_losses(train_losses, val_losses)

    # Test the model on the test dataset, get visualizations and print accuracy
    test_model(device, test_loader)

    print('Finished Training and Testing')

if __name__ == '__main__':
    main()
