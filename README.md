
## Author gender classification from text.

### Use at own risk, not well supported/documented project.

Trained on Reddit posts from r/AskMen and r/AskWomen. If I can say so myself, a clever, but abeit lazy way to get labelled data. Training was done on posts directly from those two subreddits, but this introduces its own set of biases. Maybe women who post on r/AskWomen write in a unique style inside of the subreddit, but not outside of it. To rectify this, you could instead find "women" users from the r/AskWomen, but look at their posts outside of r/AskWomen. Ideally, in a subreddit both men and women visit like r/AskReddit. 

I took ~100K posts from /r/AskMen and /r/AskWomen. 60% was used to train, and the other 40% was used for testing. NO posts outside of these two subreddits was used. The accuracy rate must be further investigated for real world data. Do not trust this model. 

|characters|accuracy|examples|
|----|--------|--------|
|below 100|65.24%|48650|
|100 to 200|75.81%|5727|
|200 to 300|73.84%|1246|
|300 to 500|74.55%|605|
|500 to 1000|69.43%|193|
|above 1000|90.48%|21|

## Use 
1. Install [pipenv](https://github.com/pypa/pipenv) and learn how to use it. 
1. Download required dependencies

    `pipenv install` to download required dependencies. 
1. Predict gender from piping in a text file. This should print out a 0 to 1 value. Male if above 0.5, otherwise female. You could consider the 0.4 to 0.6 range as inconclusive. 

    `cat some_text.txt | pipenv run python3 predict.py`

## Train your own model (not required). 
1. Install required developer dependencies. 
    
    `pipenv install --dev`
1. Run `pipenv run python3 download.py` to download Reddit posts using the PushShift API. This goes on forever until your interrupt the process. I recommend around ~130k posts. The posts are saved as `training_posts.csv`, `prelim_posts.csv`, and `testing_posts.csv` inside of the `data/` folder. The CSV's will have columns for gender, author, and body. 
1. Run `pipenv run python3 generate_weights.py` to calculate token weights (or the masculinity score). This looks through the `data/prelim_posts.csv` to generate a map of words with an associated masculinity score of -1 (feminine) to 1 (masculine) Eg. The token "wife" and "girlfriend" is used much more by men than women, so it will have `> 0` gender score. This is also done for n-pos (n = 5), which is a n-gram of parts of speech. They are saved to `data/npos_weights.json` and `data/token_weights.json`. Both are used for calculating the final features for the model.
1. Run `pipenv run python3 transform.py` to transform the training and testing posts into training data (features and labels). It will be saved as `data/training_data.npy` and `data/testing_data.npy`. 
1. Run `pipenv run python3 generate_model.py` to train the model and to save the trained model weights and biases to `data/model_weights.json` and `data/model_biases.json`.
1. Predict gender by piping in a text file.
    `cat some_text.txt | pipenv run python3 predict.py`
