import csv 
import helper
import numpy as np 
import features
from datetime import datetime

chunks = []

def collect():
  global chunks
  i = 0
  with open("data/posts.csv", "r") as f:
    for post in csv.DictReader(f, fieldnames=["gender", "author", "body"]):
      if i % 100 == 0:
        print(i)
      i += 1
      # quick tokenize func to ensure we don't divide by 0. 
      if len(helper.tokenize(post["body"])) == 0:
        continue
      x = features.get_features(post["body"])
      y = [post["gender"] == "male"]
      chunks.append(y + x)

  chunks = np.array(chunks)
  np.random.shuffle(chunks)
  np.save("data/chunks.npy", chunks)

try:
  collect()
except KeyboardInterrupt:
  # save if we interrupt early
  np.save(f"data/chunks_{round(datetime.now().timestamp())}.npy", chunks)

