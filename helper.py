import numpy as np
import requests
import json 
import re 
import math

def extract_npos(pos_list, n):
  npos = []
  cache = []
  for tag in pos_list:
    cache.append(tag)
    o = -1 
    while o >= -min(n, len(cache)):
      npos.append(" ".join(cache[o:]))
      o -= 1
  return npos


def tokenize(text):
  text = text.replace("-", "")
  text = text.replace("'", "")
  tokens = re.sub("[^a-zA-Z]", ' ', text.lower()).strip().split(" ")
  return [v for v in tokens if len(v) > 1 or v == "i"]


search_api = f"https://api.pushshift.io/reddit/comment/search"
keys = {

}

def get_comments(subreddits, authors=[], key=None, size=500):
  fb = requests.get(search_api, {
    "subreddit": ",".join(subreddits),
    "author": ",".join(authors),
    "before": keys.get(key) or "",
    "size": size,
    "sort": "desc"
  })

  if not fb.ok:
    raise Exception("Response not OK.")

  comments = fb.json()["data"]
  if len(comments) > 0 and key:
    keys[key] = comments[-1]["created_utc"]

  return [{
    "author": c["author"],
    "subreddit": c["subreddit"].lower(),
    "created": c["created_utc"],
    "body": c["body"],
    "id": c["id"]
  } for c in comments]


def contains_blacklist(text, blacklist):
  for v in blacklist:
    if text.find(v) != -1:
      return True
  return False


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


# takes list and divides it into x sections.
def chunk(arr, section_count):
  out = []
  start = 0
  chunk_size = math.floor(len(arr) / section_count)
  for i in range(section_count):
    lb = i * chunk_size
    rb = (i + 1) * chunk_size 
    
    # if last, get remaining.
    if i == (section_count - 1):
      rb = None 

    alpha = arr[lb:rb]  
    out.append(alpha)

  return out

# turns table into list, sort it via sort_key, and subdivides into buckets. 
def bucketize(table, sort_key, bucket_count):
  list_of_items = list(table.items())
  list_of_items.sort(key=sort_key)
  chunks = chunk(list_of_items, bucket_count)
  return [dict(chunk) for chunk in chunks]
