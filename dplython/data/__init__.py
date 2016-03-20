import os

import pandas as pd

# from dplython import *

root = os.path.abspath(os.path.dirname(__file__))
# diamonds = DplyFrame(pd.read_csv(os.path.join(root, "diamonds.csv")))
print os.path.join(root, "diamonds.csv")
diamonds = pd.read_csv(os.path.join(root, "diamonds.csv"))