import csv 
import helper
import numpy as np 
import features
from datetime import datetime

# transform data/<entry>.csv into data/<output>.npy
def transform(entry, output):
  chunks = []
  try:
    i = 0
    with open(entry, "r") as f:
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

    # TODO: Try to balance data sheet somewhere before.
    men = chunks[chunks[:, 0] == 1]
    women = chunks[chunks[:, 0] == 0]
    maxx = min(len(men), len(women))
    chunks = np.vstack([men[:maxx], women[:maxx]])

    np.save(output, chunks)
  except KeyboardInterrupt:
    # get balanced datasheet. 
    men = chunks[chunks[:, 0] == 1]
    women = chunks[chunks[:, 0] == 0]
    maxx = min(len(men), len(women))
    chunks = np.vstack([men[:maxx], women[:maxx]])

    timestamp = datetime.now().timestamp()
    np.save(f"{output}_{timestamp}", chunks)

  

# get training posts
transform("data/training_posts.csv", "data/training_data.npy")

# get testing posts
transform("data/testing_posts.csv", "data/testing_data.npy")