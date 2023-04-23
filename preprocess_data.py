import pandas as pd

# Read data from the csv file (from the location it is stored)
data = pd.read_csv(r'data/wikihowAll.csv')
data = data.astype(str)
rows, columns = data.shape

# Get the list of train, valid and test articles
all_train = open("data/all_train.txt").read().splitlines()
all_valid = open("data/all_val.txt").read().splitlines()
all_test = open("data/all_test.txt").read().splitlines()

# Threshold to remove short articles with long summaries as well as articles with no summary
data = data.where(data['headline'].str.len() < 0.75 * data['text'].str.len())

# Remove extra commas in abstracts
data = data.rename(columns={'headline': 'summary', 'text': 'article'})
data['summary'] = data['summary'].str.replace(".,",".")
data['summary'] = data['summary'].str.replace("\n,","\n")

# Remove extra commas in articles
data['article'] = data['article'].str.replace(r'[.]+[\n]+[,]',".\n", regex=True)
data['article'] = data['article'].str.replace(".,",".")
data['article'] = data['article'].str.replace(".;\n, ",".\n")

# Remove spaces in title for comparison
data['title'] = data['title'].str.replace(" ", "")

# Split data into train, valid and test
df_train = data[data['title'].isin(all_train)][['article', 'summary']]
df_valid = data[data['title'].isin(all_valid)][['article', 'summary']]
df_test = data[data['title'].isin(all_test)][['article', 'summary']]

# Save data
df_train.to_csv('data/train_temp.csv',encoding='utf-8-sig',index=False) 
df_valid.to_csv('data/valid_temp.csv', encoding="utf-8-sig",index=False) 
df_test.to_csv('data/test_temp.csv', encoding="utf-8-sig",index=False) 