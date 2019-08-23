import numpy as np 
from features import extract_features
from helper import sigmoid
import sys

weights = np.load("data/model_weights.npy")
biases = np.load("data/model_biases.npy")

def predict(text):
  return sigmoid(np.dot(extract_features(text), weights) + biases) 

text = "\n".join(sys.stdin.readlines()).strip() 
try:
  print(predict(text))
  exit(0)
except Exception as e:
  print(e)
  exit("-1")