
## Author gender classification from text.

### Use at own risk, not well supported/documented project.

Trained on Reddit posts from r/AskMen and r/AskWomen. If I can say so myself, a clever, but abeit lazy way to get labelled data. Training was done on posts directly from those two subreddits, but this introduces its own set of biases. Maybe women who post on r/AskWomen write in a unique style inside of the subreddit, but not outside of it. To rectify this, you could instead find "women" users from the r/AskWomen, but look at their posts outside of r/AskWomen. Ideally, in a subreddit both men and women visit like r/AskReddit. 

The accuracy rate must be further investigated for real world data. 

|length|accuracy|examples|
|----|--------|--------|
|< 250|67.56%|48481|
|200 to 500|66.02%|30715|
|500 to 1000|69.22%|13600|
|1000 to 2000|72.99%|3654|
|> 2000|76.96%|599|
|-|-|-|
|male below 250|65.98%|23527|
|male 200 to 500|65.2%|15275|
|male 500 to 1000|66.51%|6346|
|male 1000 to 2000|69.99%|1656|
|male above 2000|73.08%|286|
|-|-|-|
|female below 250|69.06%|24954|
|female 200 to 500|66.83%|15440|
|female 500 to 1000|71.59%|7254|
|female 1000 to 2000|75.48%|1998|
|female above 2000|80.51%|313|

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
1. `pipenv run python3 download.py` to download Reddit posts using the PushShift API. This goes on forever until your interrupt the process. I recommend around ~200k posts. The posts are saved to `data.db`  using sqlite3 under a "posts" table. 
1. Run `pipenv run python3 transform.py` to transform the posts into training data. Output will be stored in `data.db` under the `examples` table. 
1. Run `pipenv run python3 generate_model.py` to train and test the model. The model weights will be saved to `data/model_weights.json` and `data/model_biases.json`.
1. Predict gender by piping in a text file.
    `cat some_text.txt | pipenv run python3 predict.py`
