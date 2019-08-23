import json
import helper
import re 
from textblob.en.taggers import PatternTagger
from textblob.tokenizers import WordTokenizer

tk = WordTokenizer()
tagger = PatternTagger()

BUCKET_COUNT = 5

with open("data/npos_scores.json", "r") as f:
  npos_buckets = helper.bucketize(json.loads(f.read()), lambda x: x[1]["male"] + x[1]["female"], BUCKET_COUNT)

with open("data/ntoken_scores.json", "r") as f:
  ntoken_buckets = helper.bucketize(json.loads(f.read()), lambda x: x[1]["male"] + x[1]["female"], BUCKET_COUNT)

def get_total_scores(values, buckets, min_n, max_n):
  scores = []
  for score_table in buckets:
    scores.append(
      sum([score_table[v]["score"] for v in values if v in score_table and v.count(" ") >= min_n and v.count(" ") < max_n])
    )
  return scores 
  


def extract_features(text, skip_adv=False):

  # remove non-english and standard characters.
  text = text.strip()
  text = re.sub("[^a-zA-Z0-9\.,!\(\)\s;:\-]\"'", "", text)

  tokens = tk.tokenize(text.lower())
  pos_list = [co[1] for co in tagger.tag(text)]

  npos = helper.extract_ngrams(pos_list, 5)
  ntokens = helper.extract_ngrams(tokens, 5)

  return \
    get_total_scores(npos, npos_buckets, 1, 6) + \
    get_total_scores(ntokens, ntoken_buckets, 1, 6)