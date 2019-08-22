import helper
import numpy as np 
import features
from datetime import datetime
import sqlite3 
import json 

conn = sqlite3.connect("data.db")
c = conn.cursor()

# ensure table "examples" exists
try:
  c.execute("""CREATE TABLE examples (
    post_id text NOT NULL UNIQUE, 
    male INTEGER NOT NULL,
    x TEXT NOT NULL
  );""")
except sqlite3.OperationalError as e:
  if str(e) != "table examples already exists":
    raise e

male_count = 0
female_count = 0

try:
  index = -1
  while True:
    index += 1
    if (male_count + female_count) % 100 == 0:
      conn.commit()
      print(f"m: {male_count}, f: {female_count}")

    c.execute("SELECT id, male, body FROM posts WHERE length(body) > 250 ORDER BY ROWID DESC LIMIT ? OFFSET ?;", (500, index * 500))
    posts = c.fetchall()
    if len(posts) == 0:
      print(f"No more posts for {gender}.")
      conn.commit()
      exit()
    for post in posts:
      post_id = post[0]
      is_male = post[1]
      body = post[2] 

      x = features.extract_features(body)
      c.execute("INSERT INTO examples VALUES (?, ?, ?);", (post_id, is_male, json.dumps(x)))
      if is_male:
        male_count += 1
      else:
        female_count += 1
except KeyboardInterrupt:
  conn.commit()
    
