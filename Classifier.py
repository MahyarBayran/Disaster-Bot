"""
@author: Tristan Sparks
@author: Mahyar Bayran
"""
'''
Find movies that are similar in scripts to the original movie
'''

import os
import string
import math

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer

from nltk.util import ngrams

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
    
def ROUGE(A, B):
    # A: list of reference n-grams
    # B: list of system n-grams
    count = 0
    for bi in A:
        if bi in B:
            count += 1
    return count/len(A)

def Perplexity(A, B):
    # A: list of reference n-grams
    # B: list of system n-grams
    voc = set(A)
    voc = voc.union(set(B))
    
    p = {}
    q = {}
    for w in voc:
        p[w] = 0
        q[w] = 0
    for gram in A:
        p[gram] = p[gram] + 1
    for gram in A:
        p[gram] = p[gram] / len(A)
    for gram in B:
        q[gram] = q[gram] + 1
    for gram in B:
        q[gram] = q[gram] / len(B)

    tmp = 0
    for w in voc:
        if ((p[w] != 0) & (q[w] != 0)):
            tmp += p[w]*math.log2(q[w])

    return 2**(-tmp)
    
def remove_punc(s):

    table = str.maketrans('', '', string.punctuation)
    new_s = s.translate(table)
    return new_s
    
def get_bigrams(myString):

    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(myString)
    bigrams =  list(ngrams(tokens, 2)) 
   
    for bigram_tuple in bigrams:
        x = "%s %s" % bigram_tuple
        tokens.append(x)
    
    result = [' '.join([wordnet_lemmatizer.lemmatize(w).lower() for w in x.split()]) \
              for x in tokens if x.lower() not in stopwords.words('english') and len(x) > 8]
    
    return result

def get_unigrams(myString):

    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(myString)
    
    result = [' '.join([wordnet_lemmatizer.lemmatize(w).lower() for w in x.split()]) \
              for x in tokens if x.lower() not in stopwords.words('english') and len(x) > 8]
    return result

def find_similar_scripts(method, n_top):
    
    print("finding scripts")
    ### clean the original and extract bigrams
    file = open('corpus.txt', 'r')
    corpus = file.readlines()
    file.close()

    print("done reading files")
    unigrams_corpus = []
    bigrams_corpus = []
    for line in corpus:
        tmp = line.find(':')
        text = line[tmp+2:].strip()
        text = remove_punc(text)
        unigrams_corpus += get_unigrams(text)
        bigrams_corpus += get_bigrams(text)
        both_corpus = unigrams_corpus + bigrams_corpus
    ### load other scripts and do the same
    
    names = os.listdir('cleanedSRT')
    scores = {}
    sorted_scores = []
    sorted_scores += names
    
    for name in names:
        print('checking: '+ name + '    ...')
        file = open('cleanedSRT/'+name, 'r')
        # there is one line only
        text = file.readline()
        text = remove_punc(text) #already did in cleansrt.py
        bigrams_ex = get_bigrams(text)
        unigrams_ex = get_unigrams(text)
        both_ex = unigrams_ex + bigrams_ex
        if method == 'ROUGE':
            #score_uni = ROUGE(unigrams_corpus, unigrams_ex)
            #score_bi = ROUGE(bigrams_corpus, bigrams_ex)
            score_both = ROUGE(both_corpus, both_ex)
            scores[name] = score_both
            #print('---ROUGE Score: '+ str(score_both))
            #if score_both >= threshold:
            #    chosen_ones.append(name)
        if method == 'PERPLEXITY':
            #scores[name] = Perplexity(bigrams_corpus, bigrams_ex)
            #scores[name] = Perplexity(unigrams_corpus, unigrams_ex)
            scores[name] = Perplexity(both_corpus, both_ex)
            #print('---Perplexity Score: '+ str(scores[name]))
            
    # sort filenames by their scores
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_scores

#
# choose the method
#method1 ='ROUGE'
#method2 = 'PERPLEXITY'
#n_top = 10
#
#print(' \nROUGE : \n' )
#print(find_similar_scripts(method1, n_top))
#print(' \nPerplexity : \n' )
#print(find_similar_scripts(method2, n_top))
