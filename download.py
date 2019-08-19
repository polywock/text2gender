
import helper
import numpy as np
import csv 
import os 
import random

# used to track comment ids to avoid repeats.
ids = []

# used to track progress. 
skip_count = 0
blacklist_count = 0
saved_count = 0

# comments below this threshold won't be included.
COMMENT_LENGTH_MIN = 250

BLACKLIST_TERMS = ["http", "removed", "submission"]
AUTHOR_BLACKLIST = ["AutoModerator"]

with open("data/training_posts.csv", "w+") as f1:
  with open("data/testing_posts.csv", "w+") as f2:
     with open("data/prelim_posts.csv", "w+") as f3:
      train_writer = csv.writer(f1)
      test_writer = csv.writer(f2)
      prelim_writer = csv.writer(f3)
      while True:
        print(saved_count, blacklist_count, skip_count)
        for gender in ["male", "female"]:
          subreddit = "askmen" if gender == "male" else "askwomen" 
          posts = helper.get_comments([subreddit], [], subreddit, 500)
          for post in posts:

            # skip comments we've seen. 
            if post["id"] in ids:
              skip_count += 1
              continue
            
            # skip if body contains blacklisted terms. 
            if helper.contains_blacklist(post["body"].lower(), BLACKLIST_TERMS):
              blacklist_count += 1
              continue

            # skip if author contains blacklisted substr. 
            if helper.contains_blacklist(post["author"], AUTHOR_BLACKLIST):
              blacklist_count += 1
              continue

            # skip if body length is below threshold. 
            if len(post["body"]) < COMMENT_LENGTH_MIN:
              blacklist_count += 1
              continue

            rand = random.random()
            row = [gender, post["author"], post["body"]]
            if rand < 0.33:
              prelim_writer.writerow(row)
            elif rand < 0.66:
              test_writer.writerow(row)
            else:
              train_writer.writerow(row)
            saved_count += 1
            ids.append(post["id"])

              





