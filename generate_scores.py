
from nltk import word_tokenize, pos_tag
import csv 
import re 
import helper
import json
import os 
import sqlite3

token_freq = {}
npos_freq = {}

conn = sqlite3.connect("data.db")
c = conn.cursor()

TOKEN_USAGE_MINIMUM = 50 
NPOS_USAGE_MINIMUM = 15
MAX_NPOS = 5 # Like n-gram, but n-pos. 
POST_CHAR_LIMIT = 100
POSTS_PER_CYCLE = 500

# used to track progress
posts_processed = 0

# cycle through our posts.
# create masculinity scores for npos & tokens. 
def gather_freq():
  global posts_processed, token_freq, npos_freq
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
        # tokenize 
        tokens = word_tokenize(post[0])
        
        # turn tokens into POS tags. 
        tags = [co[1] for co in pos_tag(tokens)]

        # Extract n-grams from it 
        all_npos = helper.extract_npos(tags, 5)

        # lowercase after feeding it to get_all_npos, because nltk's pos-tag uses casing to determine proper nouns, etc. 
        # but we do not want our tokens to be case sensitive to avoid redundant variants. Eg. "she" and "She".
        tokens = [token.lower() for token in tokens]

        for token in tokens:
          token_freq[token] = token_freq.get(token, {"male": 0, "female": 0})
          token_freq[token][gender] += 1

        for n_pos in all_npos:
          npos_freq[n_pos] = npos_freq.get(n_pos, {"male": 0, "female": 0})
          npos_freq[n_pos][gender] += 1

        posts_processed += 1
        if posts_processed % 100 == 0:
          print(posts_processed)

try:
  gather_freq()
except KeyboardInterrupt:
  pass

print(f"{len(token_freq)} tokens gathered.")
print(f"{len(npos_freq)} npos gathered.")

# Equalize token_freq to acount for quantity differences. 
male_count = sum([token_freq[k]["male"] for k in token_freq])
female_count = sum([token_freq[k]["female"] for k in token_freq])
equalizer = male_count / female_count

for key in token_freq:
  token_freq[key] = {
    "male": token_freq[key]["male"],
    "female": round(token_freq[key]["female"] * equalizer)
  }

# Equalize npos_freq to acount for quantity differences. 
male_count = sum([npos_freq[k]["male"] for k in npos_freq])
female_count = sum([npos_freq[k]["female"] for k in npos_freq])
equalizer = male_count / female_count
for key in npos_freq:
  npos_freq[key] = {
    "male": npos_freq[key]["male"],
    "female": round(npos_freq[key]["female"] * equalizer)
  }


# remove values with low total count. 
token_freq = {k: token_freq[k] for k in token_freq if token_freq[k]["male"] > TOKEN_USAGE_MINIMUM and token_freq[k]["female"] > TOKEN_USAGE_MINIMUM}
npos_freq = {k: npos_freq[k] for k in npos_freq if npos_freq[k]["male"] > NPOS_USAGE_MINIMUM and npos_freq[k]["female"] > NPOS_USAGE_MINIMUM}

print(f"{len(token_freq)} tokens kept.")
print(f"{len(npos_freq)} npos kept.")

for key in token_freq:
  token_freq[key]["score"] = token_freq[key]["male"] / (token_freq[key]["male"] + token_freq[key]["female"]) * 2 - 1

for key in npos_freq:
  npos_freq[key]["score"] = npos_freq[key]["male"] / (npos_freq[key]["male"] + npos_freq[key]["female"]) * 2 - 1

os.makedirs("data", exist_ok=True)

with open("data/token_scores.json", "w+") as f:
  f.write(json.dumps(token_freq))

with open("data/npos_scores.json", "w+") as f:
  f.write(json.dumps(npos_freq))