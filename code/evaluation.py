import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys

def main(results_filepath, figure_filepath):
   results = pd.read_csv(results_filepath)
   results.drop(columns=["Epoch"], inplace=True)
   results.plot()
   plt.savefig(figure_filepath)
   plt.show()


if __name__ == "__main__":
   if len(sys.argv) >= 2:
      results_filepath = sys.argv[-2]
      figure_filepath = sys.argv[-1]
      main(results_filepath, figure_filepath)
   else:
      print("error: no filepaths given")