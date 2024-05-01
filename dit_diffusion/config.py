import torch 

IMG_SIZE = 28
PATCH_SIZE = 4
CHANNEL = 1
EMB_SIZE = 64
LABEL_NUM = 10
DIT_NUM = 3
HEAD = 4


T = 1000

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCH = 500
BATCH_SIZE = 300