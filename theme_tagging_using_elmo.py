# -*- coding: utf-8 -*-

# Importing Libraries


import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import string
import re, os

## loading ELMO model from Tensorflow Hub
elmo = hub.KerasLayer("https://tfhub.dev/google/elmo/3",trainable=False,signature="default",signature_outputs_as_dict=True)

"""# User defined functions"""

## Creating elmo vector
def elmo_vectors(x):
  return np.mean(np.array(elmo(np.array(x))['elmo']),axis=1)

## text cleaning
def clean(text):
  text=text.lower()+' '
  text=text.translate(str.maketrans('', '', string.punctuation))
  text=re.sub('\\b(i|for|years|than|have|been|at|full|time|year|a)\\W',' ',text) # removing common keywords which are irrelevant
  text=re.sub(' +', ' ', text)
  return text.strip()

## extracting top 2 most similar sentence from each theme and taking mean of cosine scores of these 2 sentences
def top_n(vec,n):
  return np.sort(vec[0])[-n:].mean()

def theme_tagging(df,themes_list,theme_elmo):
  df['Sent'] = df.Employee_Reviews.apply(clean)
  df['Tag_theme'] = ''
  df['Theme_Score'] = ''
  print(len(df))
  df.reset_index(drop=True,inplace=True)
  for sent in range(len(df)):
      try:
        temp=elmo_vectors([str(df.loc[sent,'Sent'])])

        l2=[]
        for i in themes_list:
          cs=cosine_similarity(temp,theme_elmo[i])
          l2.append(top_n(cs,2))

        df.loc[sent,'Tag_theme']=themes_list[np.array(l2).argmax()]
        df.loc[sent,'Theme_Score']=max(l2)

      except:
        df.loc[sent,'Theme_Score']=0

  df = df.loc[df['Tag_theme'] != '']
  return df

"""# Reading the theme sentences"""

theme_sent = pd.read_excel(os.getcwd() + r'/theme sentences v4.xlsx')
theme_sent.Sentence=theme_sent.Sentence.apply(clean)

theme_sent.head()

# creating a dictionary, keys as theme names and values as list of sentences as elmo vectors
theme_elmo={}
for i in theme_sent.Theme.unique():
  theme_elmo[i]=elmo_vectors(theme_sent[theme_sent.Theme==i].Sentence.tolist())

themes_list=theme_sent.Theme.unique()
themes_list

"""# Reading reviews data"""

data_df = pd.read_excel(os.getcwd() + r'/Sample_employee_reviews.xlsx').drop_duplicates()

data_df.head()

final_df = theme_tagging(data_df,themes_list,theme_elmo)
