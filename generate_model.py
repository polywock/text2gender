from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np 
import math
import os 
import sqlite3 
import json
import helper

TRAINING_COUNT = 30_000
TESTING_COUNT = 80_000

conn = sqlite3.connect("data.db")
c = conn.cursor()

def load_examples(offset, limit, is_male, minn=None, maxx=None):
  c.execute("SELECT male, x, body_length FROM examples WHERE male = ? ORDER BY ROWID ASC LIMIT ? OFFSET ?;", (int(is_male), limit, offset))
  examples = []
  for row in c.fetchall():
    if minn and row[2] < minn:
      continue
    if maxx and row[2] > maxx:
      continue
    x = json.loads(row[1])
    y = [row[0]]

    examples.append(y + list(x))

  return examples

m_train = load_examples(0, round(TRAINING_COUNT / 2), True)
f_train = load_examples(0, round(TRAINING_COUNT / 2), False)
train = np.concatenate([m_train, f_train], axis=0)
np.random.shuffle(train)
del m_train, f_train

# split training data into x, y.  
train_y, train_x = train[:, :1], train[:, 1:]



# define our sequential model. 
model = Sequential([
  # Dense(100, activation="relu"),
  Dense(1, activation="sigmoid", input_shape=train_x.shape[1:])
])

# compile to declare our optimization and loss. 
model.compile(
  optimizer="adam",
  loss="mse", # "binary_crossentropy",
  metrics=["accuracy"]
)

# train with the training data. 
model.fit(train_x, train_y, epochs=15)
del train

buckets = {}

# testing 
buckets["male"] = load_examples(round(TRAINING_COUNT / 2), round(TESTING_COUNT / 2), True)
buckets["female"] = load_examples(round(TRAINING_COUNT / 2), round(TESTING_COUNT / 2), False)
buckets["all"] = np.concatenate([buckets["male"], buckets["female"]], axis=0)

buckets["male < 250"] = load_examples(round(TRAINING_COUNT / 2), round(TESTING_COUNT / 2), True, 250)
buckets["male 250-500"] = load_examples(round(TRAINING_COUNT / 2), round(TESTING_COUNT / 2), True, 250, 500)
buckets["male 500-1000"] = load_examples(round(TRAINING_COUNT / 2), round(TESTING_COUNT / 2), True, 500, 1000)
buckets["male 1000-2000"] = load_examples(round(TRAINING_COUNT / 2), round(TESTING_COUNT / 2), True, 1000, 2000)
buckets["male > 2000"] = load_examples(round(TRAINING_COUNT / 2), round(TESTING_COUNT / 2), True, 2000)

buckets["female < 250"] = load_examples(round(TRAINING_COUNT / 2), round(TESTING_COUNT / 2), False, 250)
buckets["female 250-500"] = load_examples(round(TRAINING_COUNT / 2), round(TESTING_COUNT / 2), False, 250, 500)
buckets["female 500-1000"] = load_examples(round(TRAINING_COUNT / 2), round(TESTING_COUNT / 2), False, 500, 1000)
buckets["female 1000-2000"] = load_examples(round(TRAINING_COUNT / 2), round(TESTING_COUNT / 2), False, 1000, 2000)
buckets["female > 2000"] = load_examples(round(TRAINING_COUNT / 2), round(TESTING_COUNT / 2), False, 2000)

buckets["all < 250"] = np.concatenate([buckets["male < 250"], buckets["female < 250"]])
buckets["all 200-500"] = np.concatenate([buckets["male 250-500"], buckets["female 250-500"]])
buckets["all 500-1000"] = np.concatenate([buckets["male 500-1000"], buckets["female 500-1000"]])
buckets["all 1000-2000"] = np.concatenate([buckets["male 1000-2000"], buckets["female 1000-2000"]])
buckets["all > 2000"] = np.concatenate([buckets["male > 2000"], buckets["female > 2000"]])

# evaluate with all the tests. 
print("evaluation report\n---------------")
for bucket in buckets:
  arr = np.array(buckets[bucket])
  if len(arr) == 0:
    continue
  res = model.evaluate(arr[:, 1:], arr[:, :1], verbose=False)
  loss = f"{str(round(res[0], 2)).rjust(6)}"
  accuracy = f"{str(round(res[1] * 100, 2)).rjust(6)}%"
  examples = str(len(arr)).rjust(7)
  print(f"{bucket.ljust(11)} -> loss: {loss} - acc: {accuracy} - ex: {examples}")

# save weights and biases. 
weights, biases = model.layers[0].get_weights()
np.save("data/model_weights.npy", weights)
np.save("data/model_biases.npy", biases)
