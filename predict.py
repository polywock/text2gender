import numpy as np 
from features import get_features
from helper import sigmoid
import sys

weights = np.load("data/model_weights.npy")
biases = np.load("data/model_biases.npy")

def predict(text):
  return sigmoid(np.dot(get_features(text), weights) + biases) 

text = "\n".join(sys.stdin.readlines()).strip() 
try:
  print(predict(text))
  exit(0)
except Exception:
  exit("-1")