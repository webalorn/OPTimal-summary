'''
This code is used to create article and summary files from the csv file.
The output of the file will be a directory of text files representing seoarate articles and their summaries.
Each summary line starts with tag "@summary" and the article is followed by "@article".
'''
import pandas as pd
import os
import re

# read data from the csv file (from the location it is stored)
Data = pd.read_csv(r'data/wikihowAll.csv')
Data = Data.astype(str)
rows, columns = Data.shape

# get the list of training, validation and testing articles
all_train = open("data/all_train.txt").read().splitlines()
all_val = open("data/all_val.txt").read().splitlines()
all_test = open("data/all_test.txt").read().splitlines()

# datasets to store the articles and their summary in
df_train = pd.DataFrame(columns = ['article', 'summary'])
df_val = pd.DataFrame(columns = ['article', 'summary'])
df_test = pd.DataFrame(columns = ['article', 'summary'])

# create a file to record the file names. This can be later used to divide the dataset in train/dev/test sets
title_file = open('titles.txt', 'wb')

# The path where the articles are to be saved
'''path = "articles"
if not os.path.exists(path): os.makedirs(path)'''

# go over all the articles in the data file
for row in range(rows):
    abstract = Data.loc[row,'headline']      # headline is the column representing the summary sentences
    article = Data.loc[row,'text']           # text is the column representing the article
    title = Data.loc[row,'title']

    if row >100 :
        break

    #  a threshold is used to remove short articles with long summaries as well as articles with no summary
    if len(abstract) < (0.75*len(article)):
        
        # remove extra commas in abstracts
        abstract = abstract.replace(".,",".")
        abstract = abstract.replace("\n,","\n")
        #abstract = abstract.encode('utf-8')

        # remove extra commas in articles
        article = re.sub(r'[.]+[\n]+[,]',".\n", article)
        article = article.replace(".,",".")
        article = article.replace(".;\n, ",".\n")
        #article = article.encode('utf-8')

        if title.replace(" ", "") in all_train :
            df_new_row = pd.DataFrame([{ 'article': article, 'summary': abstract }])
            df_train = pd.concat([df_train, df_new_row])
            #df_train.append({'article' : article, 'summary' : abstract})
        if title.replace(" ", "") in all_val :
            df_new_row = pd.DataFrame([{ 'article': article, 'summary': abstract }])
            df_val = pd.concat([df_val, df_new_row])
            #df_val.append({'article' : article, 'summary' : abstract})
        if title.replace(" ", "") in all_test :
            df_new_row = pd.DataFrame([{ 'article': article, 'summary': abstract }])
            df_test = pd.concat([df_test, df_new_row])
            #df_test.append({'article' : article, 'summary' : abstract})

        # a temporary file is created to initially write the summary, it is later used to separate the sentences of the summary
        '''with open('temporaryFile.txt','wb') as t:
            t.write(abstract)
        
        # file names are created using the alphanumeric characters from the article titles.
        # they are stored in a separate text file.
        filename = Data.loc[row,'title']
        filename = "".join(x for x in filename if x.isalnum())
        filename1 = filename + '.txt'
        filename = filename.encode('utf-8')
        title_file.write(filename+b'\n')

        
        with open(path+'/'+filename1,'wb') as f:
            # summary sentences will first be written into the file in separate lines
            with open('temporaryFile.txt','r', encoding="utf8") as t:
                for line in t:
                    line=line.lower()
                    if line != "\n" and line != "\t" and line != " ":
                        f.write(b'@summary'+b'\n')
                        f.write(line.encode('utf-8'))
                        f.write(b'\n')
                    
            # finally the article is written to the file
            f.write(b'@article' + b'\n')    
            f.write(article)'''

df_train.to_csv('data/train.csv',encoding='utf-8-sig',index=False) 
df_val.to_csv('data/val.csv', encoding="utf-8-sig",index=False) 
df_test.to_csv('data/test.csv', encoding="utf-8-sig",index=False) 

title_file.close()