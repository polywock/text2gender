from nltk import word_tokenize, pos_tag
import json
import helper
import numpy as np
import re 
import math


BUCKET_COUNT = 10

with open("data/npos_scores.json", "rb") as f:
  npos_buckets = helper.bucketize(json.loads(f.read()), lambda x: x[1]["male"] + x[1]["female"], BUCKET_COUNT)

with open("data/token_scores.json", "rb") as f:
  token_buckets = helper.bucketize(json.loads(f.read()), lambda x: x[1]["male"] + x[1]["female"], BUCKET_COUNT)

def get_total_scores(values, buckets):
  scores = []
  for score_table in buckets:
    scores.append(
      sum([score_table[v]["score"] for v in values if v in score_table])
    )
  return scores 


def extract_features(text):

  # remove non-english and standard characters.
  text = text.strip()
  text = re.sub("[^a-zA-Z0-9\.,!\(\)\s;:\-]\"'", "", text)

  sentences = re.split("[\.!;\n]", text)
  tokens = word_tokenize(text)
  pos_list = [co[1] for co in pos_tag(tokens)]

  # lowercase tokens AFTER getting pos_list. NLTK's pos_tag uses capitalization to determine proper nouns, etc.
  tokens = [token.lower() for token in tokens]

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

  text_npos = helper.extract_npos(pos_list, 5)
  npos_scores = get_total_scores(text_npos, npos_buckets)
  token_scores = get_total_scores(tokens, token_buckets)
  
  return (
      
      npos_scores + \
      token_scores + \
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

  
      
  
