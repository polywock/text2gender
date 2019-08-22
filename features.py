import json
import helper
import numpy as np
import re 
import math
from textblob.en.taggers import PatternTagger
from textblob.tokenizers import WordTokenizer

tk = WordTokenizer()
tagger = PatternTagger()

BUCKET_COUNT = 5

with open("data/npos_scores.json", "rb") as f:
  npos_buckets = helper.bucketize(json.loads(f.read()), lambda x: x[1]["male"] + x[1]["female"], BUCKET_COUNT)

with open("data/ntoken_scores.json", "rb") as f:
  ntoken_buckets = helper.bucketize(json.loads(f.read()), lambda x: x[1]["male"] + x[1]["female"], BUCKET_COUNT)

def get_total_scores(values, buckets, min_n, max_n):
  scores = []
  for score_table in buckets:
    scores.append(
      sum([score_table[v]["score"] for v in values if v in score_table and v.count(" ") >= min_n and v.count(" ") < max_n])
    )
  return scores 
  

def extract_features(text):

  # remove non-english and standard characters.
  text = text.strip()
  text = re.sub("[^a-zA-Z0-9\.,!\(\)\s;:\-]\"'", "", text)

  sentences = re.split("[\.!;\n]", text)
  tokens = tk.tokenize(text.lower())
  pos_list = [co[1] for co in tagger.tag(text)]
  unique_tokens = set(tokens)

  avg_sentence_length = np.mean([len(v) for v in sentences]) - 40 # means M/F -> 39, 42
  token_count = len(tokens)
  unique_token_ratio = len(unique_tokens) / token_count # means M/F -> 39.55, 56.68

  small_word_count = len([v for v in tokens if len(v) <= 3]) / token_count
  medium_word_count = len([v for v in tokens if len(v) > 3 and len(v) <= 6]) / token_count
  long_word_count = len([v for v in tokens if len(v) > 6 and len(v) <= 12]) / token_count

  word_std = np.std([len(v) for v in tokens]) # 10% m|f diff. 
  max_word_length = max([len(v) for v in unique_tokens]) # 20% m|f diff.  

  occur_1 = sum([1 for v in unique_tokens if text.count(v) == 1]) / token_count
  occur_2 = sum([1 for v in unique_tokens if text.count(v) == 2]) / token_count
  occur_3 = sum([1 for v in unique_tokens if text.count(v) == 3]) / token_count
        
  npos = helper.extract_ngrams(pos_list, 5)
  ntokens = helper.extract_ngrams(tokens, 5)
  
  return (
      
      get_total_scores(npos, npos_buckets, 1, 6) + \
      get_total_scores(ntokens, ntoken_buckets, 1, 6) + \
      [
        avg_sentence_length, 
        token_count, 
        unique_token_ratio, 
        small_word_count, 
        medium_word_count, 
        long_word_count, 
        word_std, 
        max_word_length, 
        occur_1,
        occur_2,
        occur_3
    ]
  )

  
      
  
