import numpy as np
import pandas as pd
import argparse
import torch.nn as nn
import torch.optim as optim
import torch


class simpleNN(nn.Module):
   def __init__(self, input_size, hidden_size, output_size):
        super(simpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size))
   
   def forward(self, x):
      x = self.flatten(x)
      logits = self.linear_relu_stack(x)
      return logits

def split_data(data, training_ratio=0.2):
   data = data.sample(frac=1).reset_index(drop=True)  # shuffle data
   num_train = int(len(data) * training_ratio)
   dtrain = data.iloc[:num_train, :-1].values
   ltrain = data.iloc[:num_train, -1].values
   dval = data.iloc[num_train:, :-1].values
   lval = data.iloc[num_train:, -1].values
   return dtrain, ltrain, dval, lval


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train_loop(model, loss_fm, optimizer, training_dataloader, test_dataloader, num_epochs, device):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(training_dataloader, model, loss_fm, optimizer, device)
        test(test_dataloader, model, loss_fm, device)
    print("Done!")
    return model

def main(arguments):
      # model parameters
      input_size = 2
      output_size = 199
      hidden_size = arguments.hidden_size


      device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
      model = simpleNN(input_size, hidden_size, output_size, device)
      criterion = nn.CrossEntropyLoss()
      optimizer = optim.Adam(model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.decay_rate)

      print("Model initialized with the following parameters:")
      print(f"Input Size: {input_size}, Hidden Size: {hidden_size}, Output Size: {output_size}")
      print(f"Learning Rate: {arguments.learning_rate}")

      data = pd.read_csv(arguments.data_path)
      model = train_loop(model, criterion, optimizer, data, arguments.num_epochs, arguments.training_ratio)

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Train a simple neural network.")
   parser.add_argument("--hidden_size", type=int, default=128, help="Size of the hidden layer")
   parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
   parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
   parser.add_argument("--data_path", type=str, default="../datasets/dataset_addition.csv", help="Path to the training data")
   parser.add_argument("--output_file", type=str, default="../results/metrics.csv", help="File to save the trained model")
   parser.add_argument("--decay_rate", type=float, default=0.01, help="Size of the input layer")
   parser.add_argument("--training_ratio", type=float, default=0.2, help="Ratio of training data to total data")
   arguments = parser.parse_args()
   main(arguments)