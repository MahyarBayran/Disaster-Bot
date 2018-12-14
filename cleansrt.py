"""
@author: Tristan Sparks
@author: Mahyar Bayran
"""

import pysrt
import os

# Loading the Subtitle
names = os.listdir('srts')
count = 1;

for name in names:
    subs = pysrt.open('srts/'+name, encoding='iso-8859-1')
    file = open('cleanedSRT/'+name[:-4]+'.txt', 'w')

    subs.stream
    text = subs.text
    text = text.lower().split('\n')[1:]
    text = list(filter(lambda x: ('[' not in x) and ('<' not in x) and ('@' not in x) and ('(' not in x) and ('{' not in x), text))
    txt = ""
    
    for line in text:
        line.encode("ascii", "ignore")
        line = list(filter(lambda x: ord(x) < 128, line))
        line = ''.join(str(e) for e in line)
        if line.startswith('-'):
            txt += (line[1:].strip())
        else:
            txt += (line)
        txt += " "
    txt = txt.replace("''",'"')
    
    file.write(txt)
    file.close()
