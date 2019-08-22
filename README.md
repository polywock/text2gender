
## Author gender classification from text.

### Use at own risk, not well supported/documented project.

Trained on Reddit posts from r/AskMen and r/AskWomen. If I can say so myself, a clever, but abeit lazy way to get labelled data. Training was done on posts directly from those two subreddits, but this introduces its own set of biases. Maybe women who post on r/AskWomen write in a unique style inside of the subreddit, but not outside of it. To rectify this, you could instead find "women" users from the r/AskWomen, but look at their posts outside of r/AskWomen. Ideally, in a subreddit both men and women visit like r/AskReddit. 

The accuracy rate must be further investigated for real world data. 

|tokens|accuracy|examples|
|----|--------|--------|
|all|68.48%|34374|
|25 to 50|58.66%|179|
|50 to 100|67.02%|19084|
|100 to 200|69.58%|11163|
|200 to 500|72.83%|3596|
|500 to 1000|73.21%|321|
|above 1000|75.0%|28|
|-|-|-|
|male all|69.14%|17187|
|female all|67.81%|17187|

## Use 
1. Install [pipenv](https://github.com/pypa/pipenv) and learn how to use it. 
1. Download required dependencies

    `pipenv install`
1. Install required NLTK data.

    `pipenv run python3 -m textblob.download_corpora lite`

1. Predict gender from piping in a text file. This should print out a 0 to 1 value. Male if above 0.5, otherwise female. 

    `cat some_text.txt | pipenv run python3 predict.py`
## Train your own model (not required). 

1. Install required developer dependencies. (also ensure you have sqlite3 installed)

    `pipenv install --dev`

1. Install required NLTK data.

    `pipenv run python3 -m textblob.download_corpora lite`
1. `pipenv run python3 download.py` to download Reddit posts using the PushShift API. This goes on forever until your interrupt the process. I recommend around ~150k posts. The posts are saved to `data.db`  using sqlite3 under a "posts" table. 
1. Run `pipenv run python3 transform.py` to transform the posts into training data. Output will be stored in `data.db` under the `examples` table. This goes on forever until you interrupt it. 
1. Run `pipenv run python3 generate_model.py` to train and test the model. The model weights will be saved to `data/model_weights.json` and `data/model_biases.json`.
1. Predict gender by piping in a text file.
    `cat some_text.txt | pipenv run python3 predict.py`
