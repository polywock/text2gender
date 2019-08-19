from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np 
import math
import os 

# load training data. 
train = np.load("data/training_data.npy")
np.random.shuffle(train)

test = np.load("data/testing_data.npy")
np.random.shuffle(test)

# split testing data into buckets. 
test_male = test[test[:,0] == 1]
test_female = test[test[:,0] == 0]

buckets = {
  "all": test,
  "25 to 50": test[(test[:, -10] >= 25) & (test[:, -10] < 50)],
  "50 to 100": test[(test[:, -10] >= 50) & (test[:, -10] < 100)],
  "100 to 200": test[(test[:, -10] >= 100) & (test[:, -10] < 200)],
  "200 to 500": test[(test[:, -10] >= 200) & (test[:, -10] < 500)],
  "500 to 1000": test[(test[:, -10] >= 500) & (test[:, -10] < 1000)],
  "above 1000": test[test[:, -10] >= 1000],

  "male all": test_male,
  "male 25 to 50": test_male[(test_male[:, -10] >= 25) & (test_male[:, -10] < 50)],
  "male 50 to 100": test_male[(test_male[:, -10] >= 50) & (test_male[:, -10] < 100)],
  "male 100 to 200": test_male[(test_male[:, -10] >= 100) & (test_male[:, -10] < 200)],
  "male 200 to 500": test_male[(test_male[:, -10] >= 200) & (test_male[:, -10] < 500)],
  "male 500 to 1000": test_male[(test_male[:, -10] >= 500) & (test_male[:, -10] < 1000)],
  "male above 1000": test_male[test_male[:, -10] >= 1000],

  "female all": test_female,
  "female 25 to 50": test_female[(test_female[:, -10] >= 25) & (test_female[:, -10] < 50)],
  "female 50 to 100": test_female[(test_female[:, -10] >= 50) & (test_female[:, -10] < 100)],
  "female 100 to 200": test_female[(test_female[:, -10] >= 100) & (test_female[:, -10] < 200)],
  "female 200 to 500": test_female[(test_female[:, -10] >= 200) & (test_female[:, -10] < 500)],
  "female 500 to 1000": test_female[(test_female[:, -10] >= 500) & (test_female[:, -10] < 1000)],
  "female above 1000": test_female[test_female[:, -10] >= 1000],
}

# split training data into x, y.  
train_y, train_x = np.hsplit(train, [1]) 


# define our sequential model. 
model = Sequential([
  Dense(1, activation="sigmoid" , input_shape=train_x.shape[1:])
])

# compile to declare our optimization and loss. 
model.compile(
  optimizer="adam",
  loss="mse", #"binary_crossentropy",
  metrics=["accuracy"]
)

# train with the training data. 
model.fit(train_x, train_y, epochs=15)

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
