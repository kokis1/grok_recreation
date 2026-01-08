import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch
import sys

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

   dtrain = training[:, :2]
   ltrain = training[:, 2]

   dtest = testing[:, :2]
   ltest = testing[:, 2]

   return dtrain, ltrain, dtest, ltest

def training(model, optimizer, criterion, num_epochs, train_loader, test_loader):
   
   metrics = pd.DataFrame({"Epoch":[0], 
                               "train loss":[0], 
                               "train correct":[0], 
                               "test loss":[0], 
                               "test correct":[0]})
   
   train_losses = []
   train_corrects = []
   
   test_losses = []
   test_corrects = []
   
   
   for epoch in range(num_epochs):
      model.train()
      train_loss = 0
      train_correct = 0

      for inputs, labels in train_loader:
         optimizer.zero_grad()
         outputs = model(inputs)
         loss = criterion(outputs, labels)
         loss.backward()
         optimizer.step()

         train_loss += loss.item()
         preds = torch.argmax(outputs, dim=1)
         train_correct += (preds == labels).sum().item()
      
      model.eval()
      test_loss = 0
      test_correct = 0
      with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            test_correct += (preds == labels).sum().item()
      train_corrects.append(train_correct)
      test_corrects.append(test_correct)
      
      train_losses.append(train_loss)
      test_losses.append(test_loss)
      
      if epoch % 50 == 0:
         print(epoch, train_loss, test_loss)
   
   metrics = pd.DataFrame({"Epoch":np.arange(num_epochs), 
                              "train loss":train_losses, 
                              "train correct":train_corrects, 
                              "test loss":test_losses, 
                              "test correct":test_corrects})
   return metrics

def main(data_filepath="../datasets/dataset_addition.csv", results_filepath="../results/metrics_addition.csv"):
   
   # reads the data and prepares to send the results to the specified location
   data = pd.read_csv(data_filepath).to_numpy()
   
   # splits the dataset into a trianing and testing pair
   training_ratio = 0.3
   dtrain, ltrain, dtest, ltest = split_data(data, training_ratio)


   # parameters for the model
   input_size = 2
   output_size = 199
   hidden_size = 128
   decay_rate = 0.05
   num_epochs = 10000


   # data loaders for the training loop
   train_dataset = TensorDataset(
    torch.FloatTensor(dtrain),
    torch.LongTensor(ltrain)  # LongTensor for class indices
)
   data_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

   # Same for validation
   val_dataset = TensorDataset(
      torch.FloatTensor(dtest),
      torch.LongTensor(ltest)
   )
   test_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

   # initialises the model
   model = GrokNN(input_size, hidden_size, output_size)

   # specifies the loss and optimiser functions
   loss = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), weight_decay=decay_rate)

   # performs the training loop and saves the results as a pandas DataFrame
   metrics = training(model, optimizer, loss, num_epochs, data_loader, test_loader)
   
   # writes the results to a csv file
   metrics.to_csv(results_filepath, index=False, index_label=False)



if __name__ == "__main__":
   if len(sys.argv) >= 2:
      data_filepath = sys.argv[-2]
      results_filepath = sys.argv[-1]
      main(data_filepath, results_filepath)
   else:
      main()