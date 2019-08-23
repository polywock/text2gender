
## Author gender classification from text.

### Use at own risk, not well supported/documented project.

Trained on Reddit posts from r/AskMen and r/AskWomen. If I can say so myself, a clever, but abeit lazy way to get labelled data. Training was done on posts directly from those two subreddits, but this introduces its own set of biases. Maybe women who post on r/AskWomen write in a unique style inside of the subreddit, but not outside of it. To rectify this, you could instead find "women" users from the r/AskWomen, but look at their posts outside of r/AskWomen. Ideally, in a subreddit both men and women visit like r/AskReddit. 

The accuracy rate must be further investigated for real world data. 

|length|accuracy|examples|
|----|--------|--------|
|< 250|67.44%|48481|
|200 to 500|65.91%|30715|
|500 to 1000|69.21%|13600|
|1000 to 2000|72.47%|3654|
|> 2000|75.63%|599|
|-|-|-|
|male below 250|67.13%|23527|
|male 200 to 500|65.49%|15275|
|male 500 to 1000|68.72%|6346|
|male 1000 to 2000|74.03%|1656|
|male above 2000|78.67%|286|
|-|-|-|
|female below 250|67.74%|24954|
|female 200 to 500|66.33%|15440|
|female 500 to 1000|69.63%|7254|
|female 1000 to 2000|71.17%|1998|
|female above 2000|72.84%|313|

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
