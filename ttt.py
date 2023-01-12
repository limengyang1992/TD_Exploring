
import numpy as np
import os
import glob
from pprint import pprint

total_paths = glob.glob(os.path.join("/home/kunyu/exps","*/feature_block/*.npy"))

pprint(len(total_paths))