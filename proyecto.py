### Proyecto NLP ###

# Importar datos de reddit

import re

mcfc = open('C:/Users/tomas/Downloads/MCFC_comments', 'r', encoding='utf-8').read()

print(mcfc[:5000])

test = mcfc[:100000]

test = test.split('"body":')[1:]

comments_mcfc = []

for t in test:
  comment = t.split('","')
  if '[deleted]' not in comment[0]: 
    comments_mcfc.append(comment[0] )
 
    
clean_comments = [] 
for c in comments_mcfc:
  com = c.split('}')
  clean_comments.append(com[0])    


'''
r'[ }  http: ]')
  
re.sub('\n',' ',sent)

comments_mcfc = [t.split('","')[0] for t in test if '[deleted]' not in t.split('","')[0]]
clean_comments = [c.split('}')[0] for c in comments_mcfc]
'''









