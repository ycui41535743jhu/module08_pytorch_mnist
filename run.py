# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from utilities import *


def main():
    # Parse arguments inside the main function
    args = myargs()

    # Assign the parsed arguments to variables
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # Create an instance of the model & move it to the configured device (GPU/CPU)
    model = ConvNet().to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Nested track_accuracy function inside main
    def track_accuracy(model, loader):
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        model.train()
        return accuracy

    # Nested training function inside main
    def training(model, num_epochs, train_loader, criterion, optimizer):
        loss_list = []
        acc_train = []
        acc_test = []

        for epoch in range(num_epochs):
            pbar = tqdm(train_loader,
                        desc=f"Training: {epoch + 1}/{num_epochs}",
                        ncols=125,
                        leave=True)

            running_loss = []

            for i, (images, labels) in enumerate(pbar, 1):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=f"{sum(running_loss) / i:10.6f}")

            loss_list.append(sum(running_loss) / i)
            acc_train.append(track_accuracy(model, train_loader))
            acc_test.append(track_accuracy(model, test_loader))

        return model, loss_list, acc_train, acc_test

    # Training the model
    model, loss_list, acc_train, acc_test = training(model, num_epochs, train_loader, criterion, optimizer)

    # Make some plots
    print("Training is complete. Now plotting the results...")
    save_path = './figure.png'
    plot(loss_list, acc_train, acc_test, save_path)

# Ensure that the main function is executed when the script is run
if __name__ == "__main__":
    main()


'''python run.py -e 1 -b 10 -l 0.01'''
