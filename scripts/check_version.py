# check version of python and pytorch

import torch
import sys


print('__Python VERSION:', sys.version)

print('__pyTorch VERSION:', torch.__version__)

# check pyttorch lightning version
import pytorch_lightning as pl
print('__pyTorch Lightning VERSION:', pl.__version__)