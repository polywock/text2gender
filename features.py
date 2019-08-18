from nltk import word_tokenize, pos_tag
import json
import helper
import numpy as np
import re 
import math

SECTIONS = 5
npos_weights = None
token_weights = None

# load npos weights from file. 
def load_npos_weights():
  global npos_weights
  with open("data/npos_weights.json", "rb") as f:
    w = json.loads(f.read()) 
    w = list(w.items())
    w.sort(key= lambda x: x[1]["male"] + x[1]["female"])
    w = helper.chunk(w, SECTIONS) 
    for i in range(len(w)):
      w[i] = {co[0]: co[1] for co in w[i]}
    npos_weights = w
    
# load token weights from file. 
def load_token_weights():
  global token_weights
  with open("data/token_weights.json", "rb") as f:
    w = json.loads(f.read()) 
    w = list(w.items())
    w.sort(key= lambda x: x[1]["male"] + x[1]["female"])
    w = helper.chunk(w, SECTIONS) 
    for i in range(len(w)):
      w[i] = {co[0]: co[1] for co in w[i]}
    token_weights = w

def get_npos_scores(npos_list, n=None):
  global npos_weights

  # ensure weights are loaded.
  if not npos_weights:
    load_npos_weights()

  scores = [0] * SECTIONS

  for npos in npos_list:
      if n and npos.count(" ") + 1 != n:
        continue
      for section in range(SECTIONS):
        if npos in npos_weights[section]:
          scores[section] += npos_weights[section][npos]["ratio"]

  return scores

def get_token_scores(tokens):
  global token_weights

  # ensure weights from file. 
  if not token_weights:
    load_token_weights()

  scores = [0] * SECTIONS

  for token in tokens:
    for section in range(SECTIONS):
      if token in token_weights[section]:
        scores[section] += token_weights[section][token]["ratio"]

  return scores 

def get_features(body):
  sentences = re.split("[\.!;\n]", body)
  tokens = word_tokenize(body)
  tags = [co[1] for co in pos_tag(tokens)]
  unique_tokens = set(tokens)

  avg_sentence_length = np.mean([len(v) for v in sentences]) - 40 # means M/F -> 39, 42
  token_count = len(tokens)
  unique_token_ratio = len(unique_tokens) / token_count # means M/F -> 39.55, 56.68

  small_word_count = len([v for v in tokens if len(v) <= 3]) / token_count
  medium_word_count = len([v for v in tokens if len(v) > 3 and len(v) <= 6]) / token_count
  long_word_count = len([v for v in tokens if len(v) > 6 and len(v) <= 12]) / token_count

  word_std = np.std([len(v) for v in tokens]) # 10% diff 
  max_word_length = max([len(v) for v in unique_tokens]) # 20% diff 

  occur_1 = sum([1 for v in unique_tokens if body.count(v) == 1]) / token_count
  occur_2 = sum([1 for v in unique_tokens if body.count(v) == 2]) / token_count
  occur_3 = sum([1 for v in unique_tokens if body.count(v) == 3]) / token_count

  all_npos = helper.get_all_npos(tags, 5)
  
  return (
      get_token_scores(tokens) + \
      get_npos_scores(all_npos) + \
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

  
      
  
