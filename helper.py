import numpy as np
import requests
import json 
import re 
import math

def get_all_npos(tags, n):
  n_pos = []
  cache = []
  for tag in tags:
    cache.append(tag)
    o = -1 
    while o >= -min(n, len(cache)):
      n_pos.append(" ".join(cache[o:]))
      o -= 1
  return n_pos


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
    "authors": ",".join(authors),
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

def chunk(arr, sections):
  out = []
  start = 0
  chunk_size = math.floor(len(arr) / sections)
  for i in range(sections):
    lb = i * chunk_size
    rb = (i + 1) * chunk_size 
    
    # if last, get remaining.
    if i == (sections - 1):
      rb = None 

    alpha = arr[lb:rb]  
    out.append(alpha)

  return out