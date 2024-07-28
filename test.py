import numpy as np

import nltk
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()
import json
with open('words.json','r') as f:
    wordss=json.load(f)


all_words=[]
tags=[]
xy=[]
    
#nltk.download('punkt') do that also lol

def tokenise(word):
    return nltk.word_tokenize(word)

def stemmmer(word):
    return stemmer.stem(word.lower())
   

a="how long yo"
words=["WQasjkjk","environment"]
stem_words=[stemmmer(dd) for dd in words]
a=tokenise(a)
print(stem_words)
print(a)

for x in wordss['words']:
    tag=x['pattern']
    print(tag)
    



