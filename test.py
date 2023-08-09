VER = 14
LOAD_TOKENS_FROM = '/input/tf-longformer-v12'
LOAD_MODEL_FROM = '/input/tflongformerv14'
DOWNLOADED_MODEL_PATH = '../input/tf-longformer-v12'

import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import *

tokenizer = AutoTokenizer.from_pretrained('input')

MAX_LEN = 1024
targets = np.load('targets_1024.npy')
train_tokens = np.load('tokens_1024.npy')
train_attention = np.load('attention_1024.npy')
print('Loaded NER tokens')

config = AutoConfig.from_pretrained('config.json')
backbone = TFAutoModel.from_pretrained('tf_model.h5', config=config)