
import csv 
import re 
import helper
import json
import os 
import sqlite3
from textblob.en.taggers import PatternTagger
from textblob.tokenizers import WordTokenizer

tk = WordTokenizer()
tagger = PatternTagger()


keys = []
def key_to_int(key):
  try:
    return keys.index(key) 
  except ValueError:
    keys.append(key)
    return len(keys) - 1

ntoken_freq = {}
npos_freq = {}

conn = sqlite3.connect("data.db")
c = conn.cursor()

USAGE_MINIMUM = 15
MAX_NPOS = 5 # Like n-gram, but n-pos. 
NTOKENS_PURGE_THRESHOLD = 5E6
POST_CHAR_LIMIT = 100
POSTS_PER_CYCLE = 500

# used to track progress
posts_processed = 0



# cycle through our posts.
# create masculinity scores for npos & tokens. 
def gather_freq():
  global posts_processed, npos_freq, ntoken_freq
  index = 0
  while True:
    index += 1
    for gender in ["male", "female"]:
      # get the body's of 500 posts from database. 
      c.execute(f"SELECT body FROM posts WHERE length(body) > ? AND male = ? LIMIT ? OFFSET ?;", (POST_CHAR_LIMIT, int(gender == "male"), POSTS_PER_CYCLE, index * POSTS_PER_CYCLE))
      posts = c.fetchall()

      # if none left, we can exit. 
      if len(posts) == 0:
        return

      for post in posts:
        tokens = tk.tokenize(post[0].lower())
        pos_list = [co[1] for co in tagger.tag(post[0])]

        all_npos = helper.extract_ngrams(pos_list, 5)
        all_ntokens = helper.extract_ngrams([key_to_int(token) for token in tokens], 5, lambda x: tuple(x))

        for npos in all_npos:
          npos_freq[npos] = npos_freq.get(npos, {"male": 0, "female": 0})
          npos_freq[npos][gender] += 1

        for ntoken in all_ntokens:
          ntoken_freq[ntoken] = ntoken_freq.get(ntoken, {"male": 0, "female": 0})
          ntoken_freq[ntoken][gender] += 1

        if len(ntoken_freq) > NTOKENS_PURGE_THRESHOLD:
          print("NTOKEN PURGING!")
          ntoken_freq = {k: ntoken_freq[k] for k in ntoken_freq if ntoken_freq[k]["male"] > USAGE_MINIMUM and ntoken_freq[k]["female"] > USAGE_MINIMUM}
        
        posts_processed += 1
        if posts_processed % 100 == 0:
          print(posts_processed, len(npos_freq), len(ntoken_freq))

try:
  gather_freq()
except KeyboardInterrupt:
  pass


print(f"{len(npos_freq)} npos gathered.")
print(f"{len(ntoken_freq)} ntokens gathered.")

# If male total is 200, female total 400, all values for females will be multiplied by 0.5
helper.equalize(npos_freq, "male", "female")
helper.equalize(ntoken_freq, "male", "female")

# remove values with low total count. 
npos_freq = {k: npos_freq[k] for k in npos_freq if npos_freq[k]["male"] > USAGE_MINIMUM and npos_freq[k]["female"] > USAGE_MINIMUM}
ntoken_freq = {k: ntoken_freq[k] for k in ntoken_freq if ntoken_freq[k]["male"] > USAGE_MINIMUM and ntoken_freq[k]["female"] > USAGE_MINIMUM}

ntoken_freq = {" ".join([keys[v] for v in k]): ntoken_freq[k] for k in ntoken_freq}

print(f"{len(npos_freq)} npos kept.")
print(f"{len(ntoken_freq)} ntokens kept.")

for key in npos_freq:
  npos_freq[key]["score"] = npos_freq[key]["male"] / (npos_freq[key]["male"] + npos_freq[key]["female"]) * 2 - 1

for key in ntoken_freq:
  ntoken_freq[key]["score"] = ntoken_freq[key]["male"] / (ntoken_freq[key]["male"] + ntoken_freq[key]["female"]) * 2 - 1

os.makedirs("data", exist_ok=True)


with open("data/npos_scores.json", "w+") as f:
  f.write(json.dumps(npos_freq))

with open("data/ntoken_scores.json", "w+") as f:
  f.write(json.dumps(ntoken_freq))