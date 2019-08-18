from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np 
import math
import os 

# load training data. 
chunks = np.load("data/chunks.npy")

# shuffle it. 
np.random.shuffle(chunks)

# split into training and testing data. 
split_at = math.floor(chunks.shape[0] * 0.6)
test = chunks[split_at:]
train = chunks[:split_at]

# split testing data into buckets. 
buckets = {
  "all": test,
  "below 100": test[test[:, -10] < 100],
  "100 to 200": test[(test[:, -10] > 100) & (test[:, -10] < 200)],
  "200 to 300": test[(test[:, -10] > 200) & (test[:, -10] < 300)],
  "300 to 500": test[(test[:, -10] > 300) & (test[:, -10] < 500)],
  "500 to 1000": test[(test[:, -10] > 500) & (test[:, -10] < 1000)],
  "above 1000": test[test[:, -10] >= 1000] 
}

# split training data into x, y.  
train_y, train_x = np.hsplit(train, [1]) 
test_y, test_x = np.hsplit(test, [1])


# define our sequential model. 
model = Sequential([
  Dense(1, activation="sigmoid" , input_shape=train_x.shape[1:])
])

# compile to declare our optimization and loss. 
model.compile(
  optimizer="adam",
  loss="binary_crossentropy",
  metrics=["accuracy"]
)

# train with the training data. 
model.fit(train_x, train_y, epochs=30)

# evaluate with all the tests. 
print("evaluation report\n---------------")
for bucket in buckets:
  res = model.evaluate(buckets[bucket][:, 1:], buckets[bucket][:, :1], verbose=False)
  loss = f"{str(round(res[0], 2)).rjust(6)}"
  accuracy = f"{str(round(res[1] * 100, 2)).rjust(6)}%"
  examples = str(len(buckets[bucket])).rjust(7)
  print(f"{bucket.ljust(11)} -> loss: {loss} - acc: {accuracy} - ex: {examples}")

# save weights and biases. 
weights, biases = model.layers[0].get_weights()
np.save("data/model_weights.npy", weights)
np.save("data/model_biases.npy", biases)
