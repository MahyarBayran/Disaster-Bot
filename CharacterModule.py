# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:17:13 2018

@author: Tristan Sparks
@author: Mahyar Bayran
"""

from nltk.tokenize import sent_tokenize

import numpy as np 
import os.path
import string
from MarkovModelModule import MarkovModel

    
class Character:
    def __init__(self, name, firstlines=[]):
        self.name = name 
        self.listOfLines = []
        self.MM = []
        
        if firstlines != []:
            self.addToLines(firstlines)
                    
    def addToLines(self, newlines):
        if isinstance(newlines[0], list):
            for line in newlines:
                self.listOfLines.append(line)
        else:
            self.listOfLines.append(newlines) #a list of lists
            
    def BuildAMarkovModel(self):
        self.MM = MarkovModel(self.name, self.listOfLines, 2, 0.00001)

    def write_info(self):
        filename = "%s.txt"%self.name
        completeName = os.path.join(os.getcwd() + '\\Characters', filename)
        file = open(completeName, "w")
        file.write("Character's Name:\t%s\n\n\n"%self.name)
        self.MM.write_info(file)
