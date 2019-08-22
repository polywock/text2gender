from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np 
import math
import os 
import sqlite3 
import json

TRAINING_COUNT = 25_000
TESTING_COUNT = 25_000

conn = sqlite3.connect("data.db")
c = conn.cursor()

def load_examples(offset, limit, is_male):
  c.execute("SELECT male, x FROM examples WHERE male = ? ORDER BY ROWID ASC LIMIT ? OFFSET ?;", (int(is_male), limit, offset))
  examples = []
  for row in c.fetchall():
    x = np.array(json.loads(row[1]))[0:10]
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
  # Dense(10, activation="relu", input_shape=train_x.shape[1:]),
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


# testing 
m_test = load_examples(round(TRAINING_COUNT / 2), round(TESTING_COUNT / 2), True)
f_test = load_examples(round(TRAINING_COUNT / 2), round(TESTING_COUNT / 2), False)
test = np.concatenate([m_test, f_test], axis=0)
np.random.shuffle(test)

print(model.evaluate(test[:, 1:], test[:, :1]))
exit()

m_test = test[test[:, 0] == 1]
f_test = test[test[:, 0] == 0]

buckets = {
  "all": test,
  "25 to 50": test[(test[:, -10] >= 25) & (test[:, -10] < 50)],
  "50 to 100": test[(test[:, -10] >= 50) & (test[:, -10] < 100)],
  "100 to 200": test[(test[:, -10] >= 100) & (test[:, -10] < 200)],
  "200 to 500": test[(test[:, -10] >= 200) & (test[:, -10] < 500)],
  "500 to 1000": test[(test[:, -10] >= 500) & (test[:, -10] < 1000)],
  "above 1000": test[test[:, -10] >= 1000],

  "male all": m_test,
  "male 25 to 50": m_test[(m_test[:, -10] >= 25) & (m_test[:, -10] < 50)],
  "male 50 to 100": m_test[(m_test[:, -10] >= 50) & (m_test[:, -10] < 100)],
  "male 100 to 200": m_test[(m_test[:, -10] >= 100) & (m_test[:, -10] < 200)],
  "male 200 to 500": m_test[(m_test[:, -10] >= 200) & (m_test[:, -10] < 500)],
  "male 500 to 1000": m_test[(m_test[:, -10] >= 500) & (m_test[:, -10] < 1000)],
  "male above 1000": m_test[m_test[:, -10] >= 1000],

  "female all": f_test,
  "female 25 to 50": f_test[(f_test[:, -10] >= 25) & (f_test[:, -10] < 50)],
  "female 50 to 100": f_test[(f_test[:, -10] >= 50) & (f_test[:, -10] < 100)],
  "female 100 to 200": f_test[(f_test[:, -10] >= 100) & (f_test[:, -10] < 200)],
  "female 200 to 500": f_test[(f_test[:, -10] >= 200) & (f_test[:, -10] < 500)],
  "female 500 to 1000": f_test[(f_test[:, -10] >= 500) & (f_test[:, -10] < 1000)],
  "female above 1000": f_test[f_test[:, -10] >= 1000],
}


# evaluate with all the tests. 
print("evaluation report\n---------------")
for bucket in buckets:
  if len(buckets[bucket]) == 0:
    continue
  res = model.evaluate(buckets[bucket][:, 1:], buckets[bucket][:, :1], verbose=False)
  loss = f"{str(round(res[0], 2)).rjust(6)}"
  accuracy = f"{str(round(res[1] * 100, 2)).rjust(6)}%"
  examples = str(len(buckets[bucket])).rjust(7)
  print(f"{bucket.ljust(11)} -> loss: {loss} - acc: {accuracy} - ex: {examples}")

# save weights and biases. 
weights, biases = model.layers[0].get_weights()
np.save("data/model_weights.npy", weights)
np.save("data/model_biases.npy", biases)
