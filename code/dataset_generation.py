import numpy as np
import pandas as pd

filename = "../datasets/dataset_addition.csv"

two_digit_numbers = np.arange(100)

# generates the dataset
dataset = np.zeros((2, 3))
for num_1 in two_digit_numbers:
   for num_2 in two_digit_numbers:
      row = np.array((num_1, num_2, num_1+num_2), dtype=np.int16)
      dataset = np.vstack((dataset, row))

dataset = pd.DataFrame(dataset, dtype=np.int64)

dataset.to_csv(filename, index=False, index_label=False, header=False)