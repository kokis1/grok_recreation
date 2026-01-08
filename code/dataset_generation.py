import numpy as np
import pandas as pd

filename = "../datasets/dataset.csv"

two_digit_numbers = np.arange(100)

# generates the dataset
dataset = np.zeros((2, 3))
for num_1 in two_digit_numbers:
   for num_2 in two_digit_numbers:
      row = np.array((num_1, num_2, num_1+num_2))
      dataset = np.vstack((dataset, row))

dataset = pd.DataFrame(dataset)

dataset.to_csv(filename, index=False, index_label=False, header=False)