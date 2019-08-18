
from nltk import word_tokenize, pos_tag
import csv 
import re 
import helper
import json
import os 

token_freq = {}
npos_freq = {}
posts_processed = 0

# cycle through all our posts
# create frequency table for npos & tokens. 
with open("data/posts.csv") as f:
  for post in csv.DictReader(f, fieldnames=["gender", "author", "body"]):
    tokens = word_tokenize(post["body"])
    tags = [co[1] for co in pos_tag(tokens)]
    all_npos = helper.get_all_npos(tags, 5)


    for token in tokens:
      token_freq[token] = token_freq.get(token, {"male": 0, "female": 0})
      token_freq[token][post["gender"]] += 1

    for n_pos in all_npos:
      npos_freq[n_pos] = npos_freq.get(n_pos, {"male": 0, "female": 0})
      npos_freq[n_pos][post["gender"]] += 1

    posts_processed += 1
    if posts_processed % 100 == 0:
      print(posts_processed)



# equalize token_freq to acount for quantity differences. 
male_count = sum([token_freq[k]["male"] for k in token_freq])
female_count = sum([token_freq[k]["female"] for k in token_freq])
equalizer = male_count / female_count
for key in token_freq:
  token_freq[key] = {
    "male": token_freq[key]["male"],
    "female": round(token_freq[key]["female"] * equalizer)
  }

# equalize npos_freq to acount for quantity differences. 
male_count = sum([npos_freq[k]["male"] for k in npos_freq])
female_count = sum([npos_freq[k]["female"] for k in npos_freq])
equalizer = male_count / female_count
for key in npos_freq:
  npos_freq[key] = {
    "male": npos_freq[key]["male"],
    "female": round(npos_freq[key]["female"] * equalizer)
  }


# remove values with low total count. 
token_freq = {k: token_freq[k] for k in token_freq if token_freq[k]["male"] > 50 and token_freq[k]["female"] > 50}
npos_freq = {k: npos_freq[k] for k in npos_freq if npos_freq[k]["male"] > 15 and npos_freq[k]["female"] > 15}


for key in token_freq:
  token_freq[key]["ratio"] = token_freq[key]["male"] / (token_freq[key]["male"] + token_freq[key]["female"]) * 2 - 1

for key in npos_freq:
  npos_freq[key]["ratio"] = npos_freq[key]["male"] / (npos_freq[key]["male"] + npos_freq[key]["female"]) * 2 - 1

os.makedirs("data", exist_ok=True)

with open("data/token_weights.json", "w+") as f:
  f.write(json.dumps(token_freq))

with open("data/npos_weights.json", "w+") as f:
  f.write(json.dumps(npos_freq))