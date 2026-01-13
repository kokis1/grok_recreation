import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch
import argparse

class GrokNN(nn.Module):
   def __init__(self, input_size, hidden_size, output_size):
      super().__init__()

      self.dense_1 = nn.Linear(input_size, hidden_size)  # First layer
      self.dense_2 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
      self.dense_3 = nn.Linear(hidden_size, output_size)  # Output layer
      self.relu = nn.ReLU()  # Activation function
   
   def forward(self, x):
      x = self.dense_1(x)  # Pass through first layer
      x = self.relu(x)  # Apply activation
      x = self.dense_2(x)  # Second layer
      x = self.relu(x)  # Activation
      x = self.dense_3(x)  # Output layer
      return x



def split_data(data: np.ndarray, ratio: float) -> tuple[np.ndarray: 4]:
   # permutes the data
   data = np.random.permutation(data)
   length = int(data.shape[0] * ratio)
   training = data[:length]
   testing = data[length:]

   dtrain = training[:, :2] / 99
   ltrain = training[:, 2].astype(np.int64)

   dtest = testing[:, :2] / 99
   ltest = testing[:, 2].astype(np.int64)

   return torch.FloatTensor(dtrain), torch.LongTensor(ltrain), torch.FloatTensor(dtest), torch.LongTensor(ltest)

def training(model, optimizer, criterion, num_epochs, train_loader, test_loader):
   
   metrics = pd.DataFrame({"Epoch":[0], 
                               "train loss":[0], 
                               "train correct":[0], 
                               "test loss":[0], 
                               "test correct":[0]})
   
   train_losses = []
   train_accuracies = []
   
   test_losses = []
   test_accuracies = []

      # Diagnostic test
   print("\n=== DIAGNOSTIC TEST ===")
   model.eval()
   with torch.no_grad():
      # Get one batch
      sample_inputs, sample_labels = next(iter(train_loader))
      
      print(f"Input shape: {sample_inputs.shape}")
      print(f"Input dtype: {sample_inputs.dtype}")
      print(f"Sample input values: {sample_inputs[0]}")
      
      print(f"\nLabel shape: {sample_labels.shape}")
      print(f"Label dtype: {sample_labels.dtype}")
      print(f"Sample labels (first 10): {sample_labels[:10]}")
      print(f"Label range: {sample_labels.min()} to {sample_labels.max()}")
      
      # Forward pass
      outputs = model(sample_inputs)
      print(f"\nOutput shape: {outputs.shape}")
      print(f"Sample output (first 10 values): {outputs[0, :10]}")
      
      # Check predictions
      preds = torch.argmax(outputs, dim=1)
      print(f"\nPredictions (first 10): {preds[:10]}")
      print(f"True labels (first 10): {sample_labels[:10]}")
      
      # Check loss
      loss = criterion(outputs, sample_labels)
      print(f"\nInitial loss: {loss.item()}")
      print(f"Expected random loss: ~{np.log(199):.2f}")

   print("=== END DIAGNOSTIC ===\n")

   print("Epoch | train accuracy | test accuracy")

   for epoch in range(num_epochs):
      model.train()
      train_loss = 0
      train_correct = 0
      train_total = 0

      for inputs, labels in train_loader:
         optimizer.zero_grad()
         outputs = model(inputs)
         loss = criterion(outputs, labels)
         loss.backward()
         optimizer.step()
         
         train_loss += loss.item()
         preds = torch.argmax(outputs, dim=1)
         train_correct += (preds == labels).sum().item()
         train_total += labels.size(0)
      
      mean_train_loss = train_loss / len(train_loader)
      train_accuracy = train_correct / train_total
      
      model.eval()
      test_loss = 0
      test_correct = 0
      test_total = 0
      with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

      mean_test_loss = test_loss / len(test_loader)
      test_accuracy = test_correct / test_total
      
      train_accuracies.append(train_accuracy)
      test_accuracies.append(test_accuracy)
      
      train_losses.append(mean_train_loss)
      test_losses.append(mean_test_loss)
      
      if epoch % 50 == 0:
         print(f"{epoch} | {train_accuracy} | {test_accuracy}")
   
   metrics = pd.DataFrame({"Epoch":np.arange(num_epochs), 
                              "train loss":train_losses, 
                              "train accuracy":train_accuracies, 
                              "test loss":test_losses, 
                              "test accuracy":test_accuracies})
   return metrics

# Reinitialize weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main():

   parser = argparse.ArgumentParser(description="Train GrokNN model.")
   parser.add_argument("--data_filepath", default="../datasets/dataset_addition.csv", type=str, help="Path to the input CSV data file.")
   parser.add_argument("--results_filepath", default="../results/metrics.csv", type=str, help="Path to save the output CSV results file.")
   parser.add_argument("--decay_rate", default=0.01, type=float, help="How fast the model forgets its weights.")
   parser.add_argument("--learning_rate", default=0.01, type=float, help="How fast the model learns.")
   parser.add_argument("--num_epochs", default=500, type=int, help="How many epochs to train for.")
   parser.add_argument("--train_ratio", default=0.3, type=float, help="Proportion of the dataset used for training.")
   args = parser.parse_args()

   data_filepath = args.data_filepath
   results_filepath = args.results_filepath
   decay_rate = args.decay_rate
   learning_rate = args.learning_rate
   num_epochs = args.num_epochs
   training_ratio = args.train_ratio

   # reads the data and prepares to send the results to the specified location
   data = pd.read_csv(data_filepath).to_numpy()
   
   # splits the dataset into a trianing and testing pair
   dtrain, ltrain, dtest, ltest = split_data(data, training_ratio)


   # parameters for the model
   input_size = 2
   output_size = 199
   hidden_size = 128


   # data loaders for the training loop
   train_dataset = TensorDataset(
    dtrain, ltrain  # LongTensor for class indices
)
   data_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

   # Same for validation
   val_dataset = TensorDataset(
      dtest, ltest
   )
   test_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

   # initialises the model
   model = GrokNN(input_size, hidden_size, output_size)
   model.apply(init_weights)

   # specifies the loss and optimiser functions
   loss = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), weight_decay=decay_rate, lr=learning_rate)

   # performs the training loop and saves the results as a pandas DataFrame
   metrics = training(model, optimizer, loss, num_epochs, data_loader, test_loader)
   
   # writes the results to a csv file
   metrics.to_csv(results_filepath, index=False, index_label=False)



if __name__ == "__main__":
      main()