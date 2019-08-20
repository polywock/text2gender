
import helper
import numpy as np
import os 
import random
import sqlite3

conn = sqlite3.connect("data.db")
c = conn.cursor()

# Ensure posts table exists.
try:
  c.execute("""CREATE TABLE posts (
    id TEXT PRIMARY KEY, 
    created INTEGER NOT NULL, 
    male INTEGER NOT NULL, 
    author TEXT, 
    body TEXT
  );""")
  conn.commit()
except sqlite3.OperationalError as err:
  if str(err) != "table posts already exists":
    raise err


# used to track progress. 
skip_count = 0
blacklist_count = 0
saved_count = 0


BLACKLIST_TERMS = ["http", "removed", "submission"]
AUTHOR_BLACKLIST = ["AutoModerator"]


try:
  while True:
    print("saved", "blacklisted", "skipped")
    print(saved_count, blacklist_count, skip_count)
    for gender in ["male", "female"]:
      subreddit = "askmen" if gender == "male" else "askwomen" 
      posts = helper.get_comments([subreddit], [], subreddit, 500)
      for post in posts:
        
        # skip if body contains blacklisted terms. 
        if helper.contains_blacklist(post["body"].lower(), BLACKLIST_TERMS):
          blacklist_count += 1
          continue

        # skip if author contains blacklisted substr. 
        if helper.contains_blacklist(post["author"].lower(), AUTHOR_BLACKLIST):
          blacklist_count += 1
          continue

        try:
          c.execute("INSERT INTO posts VALUES (?, ?, ?, ?, ?);", (post["id"], int(post["created"]), int(gender == "male"), post["author"], post["body"]))
          saved_count += 1
          if saved_count % 500 == 0:
            conn.commit()
        except sqlite3.OperationalError as e:
          if str(e) != "UNIQUE constraint failed: posts.id":
            print(e)
except Exception:
  conn.commit()
