# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:15:46 2018

@author: Tristan Sparks
@author: Mahyar Bayran
"""
import numpy as np
import random

# for n = 3, this takes >5 minutes and approximately 4*wordcount MB of RAM
class MarkovModel:
    '''
    This is a standard markov model
    INDEX:
        name - string name
        listOfLines - a list of the lines to be markovised
        n - bigram model, markov order = n - 1
        smooth_param - smoothing parameter, by add-delta smoothing
    '''

    def __init__(self, name, listOfLines, n, smooth_param=0):
        self.name = name
        self.listOfLines = listOfLines
        self.n = n
        self.smooth_param = smooth_param
        self.states = {}
        self.wordCount = 0
        
        # initialize states, map from word to index of word in distributions for convienence
        for line in listOfLines:
            for word in line:
                if (word not in self.states.keys()):
                    self.states[word] = self.wordCount
                    self.wordCount += 1

        self.shape = [ self.wordCount for _ in range(self.n) ]
        self.initial_dist = np.zeros(self.wordCount)
        self.transition = np.zeros(self.shape)
        self.calc_initial()
        self.calc_transition()
        
    def calc_initial(self):
        for line in self.listOfLines:
            self.initial_dist[self.states[line[0]]] += 1 
           
        self.initial_dist = np.array([ elem + self.smooth_param for elem in self.initial_dist ])
        self.initial_dist /= ( len(self.initial_dist) * self.smooth_param + len(self.listOfLines) )
        
    def calc_transition(self):
        # First Order Markov
        if (self.n == 2):
            for line in self.listOfLines:
                for i in range(len(line) - (self.n - 1)):
                    self.transition[self.states[line[i]]][self.states[line[i+1]]] += 1
            for i in range(len(self.transition)):
                corpus_size = sum(self.transition[i]) # corpus size is the # pairs of states where i is the first element in the pair
                if (corpus_size > 0 or self.smooth_param > 0):
                    self.transition[i] = [ elem + self.smooth_param for elem in self.transition[i] ]
                    self.transition[i] /= ( corpus_size + self.smooth_param * self.wordCount)
        
        # Second Order Markov
        elif (self.n == 3):
            for line in self.listOfLines:
                for i in range(len(line) - (self.n - 1)):
                    self.transition[self.states[line[i]]][self.states[line[i+1]]][self.states[line[i+2]]] += 1
            for i in range(self.wordCount):
                for j in range(self.wordCount):
                    corpus_size = sum(self.transition[i][j]) # corpus size is the # triplets of states where i,j are the first elements in the triplet
                    if (corpus_size > 0 or self.smooth_param > 0):
                        self.transition[i][j] = [ elem + self.smooth_param for elem in self.transition[i][j] ]
                        self.transition[i][j] /= ( corpus_size + self.smooth_param * self.wordCount)
        else:
            print("n must be 2 or 3")
            return
        
    def write_info(self, file):
        file.write('States: \n'+ str(list(self.states.items())) +'\n\n\n')
        file.write('Initial Distributions: \n' + str(self.initial_dist.tolist()) + '\n\n\n')
        file.write('Transition Probabilities: \n' + str(self.transition.tolist() )+ '\n\n\n')
        
        
    def generate(self, word_limit=20, overflow_allowed=20):
        end_tokens = [".", "?", "!"]
        
        words = [0 for _ in range(len(self.states))]
        for elem in self.states.items():
            words[elem[1]] = elem[0]
        
        # how to pick seed words for n=3?
        first_word = random_choice(words, self.initial_dist)
        generated_sent = [first_word]
        
        if (self.n == 3):
            # for now pick second word from random line that begins with the first word
            lines = []
            for line in self.listOfLines:
                if (line[0] == first_word):
                    lines.append(line)
                    
            line = random.choice(lines)
            generated_sent.append(line[1])
        
        for i in range(self.n - 1, word_limit):
            if (self.n == 2):
                probs = self.transition[self.states[generated_sent[i-1]]]
            elif (self.n == 3):
                probs = self.transition[self.states[generated_sent[i-2]]][self.states[generated_sent[i-1]]]
            
            generated_sent.append(random_choice(words, probs))
            
        i = word_limit
        while(generated_sent[-1] not in end_tokens and len(generated_sent) <= word_limit + overflow_allowed):
            if (self.n == 2):
                probs = self.transition[self.states[generated_sent[i-1]]]
            elif (self.n == 3):
                probs = self.transition[self.states[generated_sent[i-2]]][self.states[generated_sent[i-1]]]
            
            generated_sent.append(random_choice(words, probs))
            i += 1
            
        return generated_sent
        

class WeightedComboMarkovModel:
    '''
    This model takes two markov models,
    and combines their initial and transition probability distributions
    INPUTS:
        primaryMM - the primary corpus markov model
        externalMM - the external markov model 
        weight - the weight attributed to the primary corpus (float between 0 and 1 (incl))
    '''
    def __init__(self, primaryMM, externalMM, weight):
        if (externalMM.n != primaryMM.n):
            print("primary and external corpora must be of same order")
            return

        self.primaryMM = primaryMM
        self.externalMM = externalMM
        self.n = primaryMM.n
        self.weight = weight
        self.states = {}
        self.wordCount = 0
        
        for word in primaryMM.states.keys():
            self.states[word] = self.wordCount
            self.wordCount += 1
        for word in externalMM.states.keys():
            if word not in self.states.keys():
                self.states[word] = self.wordCount
                self.wordCount += 1

        self.shape = [ self.wordCount for _ in range(self.n) ]
        self.initial_dist = np.zeros(self.wordCount)
        self.transition = np.zeros(self.shape)
        self.calc_initial()
        self.calc_transition()
        
    def calc_initial(self):
        self.initial_dist = np.zeros(len(self.states))
        for word, index in self.states.items():
            if word in self.primaryMM.states.keys():
                i1 = self.primaryMM.initial_dist[self.primaryMM.states[word]]
            else:
                i1 = 0
            if word in self.externalMM.states.keys():
                i2 = self.externalMM.initial_dist[self.externalMM.states[word]]
            else:
                i2 = 0  
            
            self.initial_dist[index] = i1 * self.weight + i2 * (1 - self.weight)
            
        
    def calc_transition(self):
        self.transition = np.zeros( (len(self.states), len(self.states)) )
        
        for word1, index1 in self.states.items():
            for word2, index2 in self.states.items():
                # First Order Markov
                if (self.n == 2):
                    if (word1 in self.primaryMM.states.keys() and word2 in self.primaryMM.states.keys()):
                        t1 = self.primaryMM.transition[self.primaryMM.states[word1]][self.primaryMM.states[word2]]
                    else:
                        t1 = 0
                    if (word1 in self.externalMM.states.keys() and word2 in self.externalMM.states.keys()):
                        t2 = self.externalMM.transition[self.externalMM.states[word1]][self.externalMM.states[word2]]
                    else:
                        t2 = 0
                    
                    
                    if (word1 not in self.primaryMM.states.keys()):
                        self.transition[index1][index2] = t2
                    elif (word1 not in self.externalMM.states.keys()):
                        self.transition[index1][index2] = t1
                    else:
                        self.transition[index1][index2] = t1 * self.weight + t2 * (1 - self.weight)
                
                # Second Order Markov
                if (self.n == 3):
                    for word3, index3 in self.states.items():
                        if (word1 in self.primaryMM.states.keys() and word2 in self.primaryMM.states.keys() and word3 in self.primaryMM.states.keys()):
                            t1 = self.primaryMM.transition[self.primaryMM.states[word1]][self.primaryMM.states[word2]][self.primaryMM.states[word3]]
                        else:
                            t1 = 0
                        if (word1 in self.externalMM.states.keys() and word2 in self.externalMM.states.keys() and word3 in self.primaryMM.states.keys()):
                            t2 = self.externalMM.transition[self.externalMM.states[word1]][self.externalMM.states[word2]][self.externalMM.states[word3]]
                        else:
                            t2 = 0
                        
                        
                        if (word1 not in self.primaryMM.states.keys() or word2 not in self.primaryMM.states.keys()):
                            self.transition[index1][index2][index3] = t2
                        elif (word1 not in self.externalMM.states.keys() or word2 not in self.externalMM.states.keys()):
                            self.transition[index1][index2][index3] = t1
                        else:
                            self.transition[index1][index2][index3] = t1 * self.weight + t2 * (1 - self.weight)
                    
    def generate(self, word_limit=20, overflow_allowed=20):
        end_tokens = [".", "?", "!"]
        
        words = [0 for _ in range(len(self.states))]
        for elem in self.states.items():
            words[elem[1]] = elem[0]
        
        # how to pick seed words for n=3?
        first_word = random_choice(words, self.initial_dist)
        generated_sent = [first_word]
        if (self.n == 3):
            # for now pick second word from random line that begins with the first word
            lines = []
            for line in self.listOfLines:
                if (line[0] == first_word):
                    lines.append(line)
                    
            line = random.choice(lines)
            generated_sent.append(line[1])
        
        for i in range(self.n - 1, word_limit):
            if (self.n == 2):
                probs = self.transition[self.states[generated_sent[i-1]]]
            elif (self.n == 3):
                probs = self.transition[self.states[generated_sent[i-2]]][self.states[generated_sent[i-1]]]
            
            generated_sent.append(random_choice(words, probs))
            
        i = word_limit
        while(generated_sent[-1] not in end_tokens and len(generated_sent) <= word_limit + overflow_allowed):
            if (self.n == 2):
                probs = self.transition[self.states[generated_sent[i-1]]]
            elif (self.n == 3):
                probs = self.transition[self.states[generated_sent[i-2]]][self.states[generated_sent[i-1]]]
            
            generated_sent.append(random_choice(words, probs))
            i += 1
            
        return generated_sent
        
class NormalizedComboMarkovModel:
    '''
    This model takes two Markov Models,
    and combines their initial and transition distributions, however, only external
    information corresponding to words in the primary corpus is used 
    INPUTS:
        primaryMM - the primary corpus markov model
        externalMM - the external markov model 
        weight - the weight attributed to the primary corpus (float between 0 and 1 (incl))
    '''
    def __init__(self, primaryMM, externalMM, weight):
        if (externalMM.n != primaryMM.n):
            print("primary and external corpora must be of same order")
            return
        
        self.primaryMM = primaryMM
        self.externalMM = externalMM
        self.weight = weight
        self.n = self.primaryMM.n
        self.states = self.primaryMM.states
        self.wordCount = len(self.states)

        self.calc_initial()
        self.calc_transition()
        
    def calc_initial(self):
        self.initial_dist = self.primaryMM.initial_dist.copy()
        
        external_weights = np.zeros(len(self.initial_dist))
        for word, index in self.states.items():
            if (word in self.externalMM.states.keys()):
                external_weights[index] = self.externalMM.initial_dist[self.externalMM.states[word]]
         
        # Normalize external info
        external_dist = softmax(external_weights)
        
        # Combine with primary info
        self.initial_dist = self.initial_dist * self.weight + external_dist * (1 - self.weight)
        
    def calc_transition(self):  
        self.transition = self.primaryMM.transition.copy()
        external_weights = np.zeros(self.primaryMM.shape)
        
        for word1, index1 in self.states.items():
            for word2, index2 in self.states.items():
                # First Order Markov
                if (self.n == 2):
                    if (word1 in self.externalMM.states.keys() and word2 in self.externalMM.states.keys()):
                        external_weights[index1][index2] = self.externalMM.transition[self.externalMM.states[word1]][self.externalMM.states[word2]]
                # Second Order Markov
                elif (self.n == 3):
                    for word3, index3 in self.states.items():
                        if (word1 in self.externalMM.states.keys() and word2 in self.externalMM.states.keys() and word3 in self.externalMM.states.keys()):
                            external_weights[index1][index2][index3] = self.externalMM.transition[self.externalMM.states[word1]][self.externalMM.states[word2]][self.externalMM.states[word3]]
        
        # Normalize external info and combine with primary
        external_dist = np.zeros(self.primaryMM.shape)
        if (self.n == 2):
            for i in range(len(self.transition)):
                external_dist[i] = softmax(external_weights[i]) #normalize
                self.transition[i] = self.transition[i] * self.weight + external_dist[i] * (1 - self.weight)
                
        elif (self.n == 3):
            for i in range(len(self.transition)):
                for j in range(len(self.transition)):
                    external_dist[i][j] = softmax(external_weights[i][j])
                    self.transition[i][j] = self.transition[i][j] * self.weight + external_dist[i][j] * (1 - self.weight)

                          
    def generate(self, word_limit=20, overflow_allowed=20):
        end_tokens = [".", "?", "!"]
        
        words = [0 for _ in range(len(self.states))]
        for elem in self.states.items():
            words[elem[1]] = elem[0]
        
        # how to pick seed words for n=3?
        first_word = random_choice(words, self.initial_dist)
        generated_sent = [first_word]
        if (self.n == 3):
            # for now pick second word from random line that begins with the first word
            lines = []
            for line in self.listOfLines:
                if (line[0] == first_word):
                    lines.append(line)
                    
            line = random.choice(lines)
            generated_sent.append(line[1])
        
        for i in range(self.n - 1, word_limit):
            if (self.n == 2):
                probs = self.transition[self.states[generated_sent[i-1]]]
            elif (self.n == 3):
                probs = self.transition[self.states[generated_sent[i-2]]][self.states[generated_sent[i-1]]]
            
            generated_sent.append(random_choice(words, probs))
            
        i = word_limit
        while(generated_sent[-1] not in end_tokens and len(generated_sent) <= word_limit + overflow_allowed):
            if (self.n == 2):
                probs = self.transition[self.states[generated_sent[i-1]]]
            elif (self.n == 3):
                probs = self.transition[self.states[generated_sent[i-2]]][self.states[generated_sent[i-1]]]
            
            generated_sent.append(random_choice(words, probs))
            i += 1
            
        return generated_sent
    
def random_choice(options, probabilities):
    d = {}
    for i in range(len(options)):
        d[options[i]] = probabilities[i]

    sorted_by_value = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    
    cumm_sorted = [0]
    for i in range(len(sorted_by_value)):
        cumm_sorted.append(cumm_sorted[-1] + sorted_by_value[i][1])
    
    random_num = random.uniform(0, 1)
    
    index = 0
    for i in range(len(sorted_by_value)):
        if ( (random_num >= cumm_sorted[i]) & (random_num <= cumm_sorted[i+1]) ):
            index = i
            break

    return sorted_by_value[index][0]
    
    
def softmax(X, theta = 1, axis = None):
    """
    Written by Nolan B Conway, https://nolanbconaway.github.io/blog/2017/softmax-numpy
    
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    
    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p