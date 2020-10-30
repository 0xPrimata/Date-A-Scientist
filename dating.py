import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import re
from sklearn.naive_bayes import MultinomialNB

regex = re.compile(r'\(.*\)$')
df = pd.read_csv('profiles.csv')

#Augmenting data
# df['is_comf'] = df['body_type'].apply([lambda x: 0 if x == 'rather not say' else 1])
# df['is_vegan'] = df['diet'].apply([lambda x: 1 if (x == 'vegan') | (x == 'strictly vegan') | (x == 'mostly vegan') else 0])
# df["drinks_code"] = df.drinks.map({"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5})
# df["smokes_code"] = df.smokes.map({"no": 0, "trying to quit": 1, "when drinking": 2, "sometimes": 3, "yes": 4})
# df["drugs_code"] = df.drugs.map({"never": 0, "sometimes": 1, "often": 2})
df['plural_code'] = df.ethnicity.apply([lambda x: 0 if (x == 'white') else 1 if (x == 'black') or (x == 'asian') or (x == 'hispanic / latin') else 2]).replace(np.nan, 0, regex=True)
df['speaks_count']  = df.speaks.str.replace('\(([^\)]+)\)', '', case=False).str.replace(' ', '').replace(np.nan, '', regex=True).str.split(",").apply([lambda x: len(x)]).replace(np.nan, 1, regex=True)
df['religion_code'] = df.religion.map({'agnosticism': 1,'other': 2,'agnosticism but not too serious about it': 1,
'agnosticism and laughing about it': 1,'catholicism but not too serious about it  ': 0,'atheism': 1,'other and laughing about it': 2,
'atheism and laughing about it': 1,'christianity': 0,'christianity but not too serious about it': 0,'other but not too serious about it':
 2,'judaism but not too serious about it': 1,'atheism but not too serious about it': 1,'catholicism': 0,'christianity and somewhat serious about it': 0,
'atheism and somewhat serious about it': 1,'other and somewhat serious about it': 2,'catholicism and laughing about it': 0,
'judaism and laughing about it': 1,'buddhism but not too serious about it': 2,'agnosticism and somewhat serious about it': 1,'judaism': 1,
'christianity and very serious about it': 0,'atheism and very serious about it': 1,'catholicism and somewhat serious about it': 0,
'other and very serious about it': 2,'buddhism and laughing about it': 2,'buddhism': 2,'christianity and laughing about it': 0,
'buddhism and somewhat serious about it': 2,'agnosticism and very serious about it': 1,'judaism and somewhat serious about it': 1,'hinduism but not too serious about it': 2,
'hinduism': 2,'catholicism and very serious about it': 0,'buddhism and very serious about it': 2,'hinduism and somewhat serious about it': 2,
'islam': 2,'hinduism and laughing about it' : 2,'islam but not too serious about it': 2,'islam and somewhat serious about it': 2,
'judaism and very serious about it': 1,'islam and laughing about it': 2,'hinduism and very serious about it': 2,'islam and very serious about it': 2}).replace(np.nan, 0, regex=True)

#original education
df['education_code'] = df['education'].map({'graduated from college/university': 1,'graduated from masters program': 2,
'working on college/university': 0,'working on masters program': 1,'graduated from two-year college': 1,
'graduated from high school': 0,'graduated from ph.d program': 3,'graduated from law school': 1,
'working on two-year college': 0,'dropped out of college/university': 0,'working on ph.d program': 2,
'college/university': 1,'graduated from space camp': 1,'dropped out of space camp': 0,
'graduated from med school': 1,'working on space camp': 0,'working on law school': 0,
'two-year college': 1,'working on med school': 0,'dropped out of two-year college': 0,
'dropped out of masters program': 2,'masters program': 2,'dropped out of ph.d program': 3,
'dropped out of high school': 0,'high school': 0,'working on high school': 0,'space camp': 1,
'ph.d program': 3,'law school': 1,'dropped out of law school': 0,'dropped out of med school': 0,'med school': 1}).replace(np.nan, 0, regex=True)

# # dropout & on the process as an intermediate group
# df['education_code'] = df['education'].map({'graduated from college/university': 3,'graduated from masters program': 5,
# 'working on college/university': 2,'working on masters program': 4,'graduated from two-year college': 3,
# 'graduated from high school': 1,'graduated from ph.d program': 7,'graduated from law school': 5,
# 'working on two-year college': 2,'dropped out of college/university': 2,'working on ph.d program': 6,
# 'college/university': 3,'graduated from space camp': 3,'dropped out of space camp': 2,
# 'graduated from med school': 3,'working on space camp': 2,'working on law school': 2,
# 'two-year college': 3,'working on med school': 2,'dropped out of two-year college': 2,
# 'dropped out of masters program': 4,'masters program': 5,'dropped out of ph.d program': 6,
# 'dropped out of high school': 0,'high school': 1,'working on high school': 0,'space camp': 3,
# 'ph.d program': 7,'law school': 3,'dropped out of law school': 2,'dropped out of med school': 2,'med school': 3}).replace(np.nan, 0, regex=True)

# grouping dropouts with completed
# df['education_code'] = df['education'].map({'graduated from college/university': 1,'graduated from masters program': 2,
# 'working on college/university': 1,'working on masters program': 1,'graduated from two-year college': 1,
# 'graduated from high school': 0,'graduated from ph.d program': 3,'graduated from law school': 1,
# 'working on two-year college': 1,'dropped out of college/university': 1,'working on ph.d program': 3,
# 'college/university': 1,'graduated from space camp': 1,'dropped out of space camp': 1,
# 'graduated from med school': 1,'working on space camp': 1,'working on law school': 1,
# 'two-year college': 1,'working on med school': 1,'dropped out of two-year college': 1,
# 'dropped out of masters program': 2,'masters program': 2,'dropped out of ph.d program': 3,
# 'dropped out of high school': 0,'high school': 0,'working on high school': 0,'space camp': 1,
# 'ph.d program': 3,'law school': 1,'dropped out of law school': 1,'dropped out of med school': 1,'med school': 1}).replace(np.nan, 0, regex=True)

features = df[['religion_code', 'speaks_count', 'education_code', 'age']]
labels = df['plural_code']

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

model = RandomForestClassifier()
model.fit(x_train, y_train)
predict = model.predict(x_test)
score = model.score(x_test, y_test)

# features = df[['drinks_code', 'smokes_code', 'drugs_code']].replace(np.nan, 0, regex=True)
# min_max_scaler = MinMaxScaler() 
# features_scaled = min_max_scaler.fit_transform(x)

#cleaning labels for is_comf/vegan
# labels = df['is_comf'].replace(np.nan, 0, regex=True)
# print(np.unique(labels, return_counts=True))

# train_x, test_x, train_y, test_y = train_test_split(features_scaled, labels, test_size=0.2)

# Logistic / Linear Regression on vegan / comf = bad prediction
# model = LogisticRegression()
# model.fit(train_x, train_y)
# predict = model.predict(test_x)
# score = model.score(test_x, test_y)

# KNeighbors on vegan / comf = bad prediction
# neigh = KNeighborsRegressor(n_neighbors = 10)
# neigh.fit(train_x, train_y)
# predict2 = neigh.predict(test_x)
# score = neigh.score(test_x, test_y)


# print(test_y.value_counts())

# print(np.unique(predict2, return_counts=True))

# precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, predict, zero_division=0)
# print(precision)
# print(recall)
# print(fbeta_score)
# print(support)

# KNeighbors attempt on language prediction
# for i in range(2, 8):
#     model = KNeighborsClassifier(n_neighbors=i).fit(x_train, y_train)
#     predict = model.predict(x_test)
#     print("kneighbors: " + str(i),  np.unique(predict, return_counts=True)) 
#     print(np.unique(y_test, return_counts=True)) 
#     print(confusion_matrix(y_test, predict))
#     print(classification_report(y_test, predict))

#KMeans attempt to blindly predict languages
# for i in range(1, 20):
#   model = KMeans(n_clusters=i).fit(x_train)
#   predict = model.predict(x_test)
#   print("clusters: " + str(i),  np.unique(predict, return_counts=True))
# print("actual: " + str(i),  np.unique(y_test, return_counts=True))

print(score)
print(str("predict"),  np.unique(predict, return_counts=True)) 
print(str("actual"), np.unique(y_test, return_counts=True)) 
print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))