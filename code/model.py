import numpy as np
import pandas as pd
import argparse
import torch.nn as nn
import torch.optim as optim
import torch


class simpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(simpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(x)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def split_data(data, training_ratio=0.2):
   data = data.sample(frac=1).reset_index(drop=True)  # shuffle data
   num_train = int(len(data) * training_ratio)
   dtrain = data.iloc[:num_train, :-1].values
   ltrain = data.iloc[:num_train, -1].values
   dval = data.iloc[num_train:, :-1].values
   lval = data.iloc[num_train:, -1].values
   return dtrain, ltrain, dval, lval


def train(model, criterion, optimizer, data, num_epochs, training_ratio):
   dtrain, ltrain, dval, lval = split_data(data, training_ratio)
   dtrain = torch.tensor(dtrain, dtype=torch.float32)
   ltrain = torch.tensor(ltrain, dtype=torch.float32)
   dval = torch.tensor(dval, dtype=torch.float32)
   lval = torch.tensor(lval, dtype=torch.float32)

   metrics = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}
   for epoch in range(num_epochs):
      
      # performs one step of backpropagation
      model.train()
      optimizer.zero_grad()
      outputs = model(dtrain)
      loss = criterion(outputs.squeeze(), ltrain)
      loss.backward()
      optimizer.step()
      
      # evaluates the model on training data
      train_loss = loss.item()
      predictions = outputs.argmax(dim=1)
      correct = (predictions == ltrain).sum().item()
      train_accuracy = correct / ltrain.size(0)

      # evaluates the model on validation data
      model.eval()
      with torch.no_grad():
         val_outputs = model(dval)
         val_loss = criterion(val_outputs.squeeze(), lval).item()
         val_predictions = val_outputs.argmax(dim=1)
         val_correct = (val_predictions == lval).sum().item()
         val_accuracy = val_correct / lval.size(0)

      metrics["train_loss"].append(train_loss)
      metrics["val_loss"].append(val_loss)
      metrics["train_accuracy"].append(train_accuracy)
      metrics["val_accuracy"].append(val_accuracy)

      if (epoch+1) % 10 == 0:
         print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')
      
      return pd.DataFrame(metrics)
def main():
      parser = argparse.ArgumentParser(description="Train a simple neural network.")
      parser.add_argument("--hidden_size", type=int, default=128, help="Size of the hidden layer")
      parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
      parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
      parser.add_argument("--data_path", type=str, default="../datasets/dataset_addition.csv", help="Path to the training data")
      parser.add_argument("--output_file", type=str, default="../results/metrics.csv", help="File to save the trained model")
      parser.add_argument("--decay_rate", type=float, default=0.01, help="Size of the input layer")
      parser.add_argument("--training_ratio", type=float, default=0.2, help="Ratio of training data to total data")
      args = parser.parse_args()

      # model parameters
      input_size = 2
      output_size = 199
      hidden_size = args.hidden_size
   
      model = simpleNN(input_size, hidden_size, output_size)
      criterion = nn.MSELoss()
      optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)

      data = pd.read_csv(args.data_path)

      metrics = train(model, criterion, optimizer, data, args.num_epochs, args.training_ratio)

      metrics.to_csv(args.output_file, index=False)
   
      print("Model initialized with the following parameters:")
      print(f"Input Size: {args.input_size}, Hidden Size: {args.hidden_size}, Output Size: {args.output_size}")
      print(f"Learning Rate: {args.learning_rate}")

if __name__ == "__main__":
   main()