#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


import pandas as pd
import json


# ### Loading of Dataset

# In[2]:


def getDataframe(file):
    with open(file,'r') as f:
        data = [json.loads(entity) for entity in f]

    return pd.DataFrame(data) 


# ### Converting to Dataframe

# In[3]:


# reviews = getDataframe('Electronics_5.json')
# metadata = getDataframe('meta_Electronics.json')


# In[4]:


# reviews


# In[5]:


# metadata


# ### Storing and Loading from pickle file

# In[6]:


import pickle


# In[7]:


# def store_dataframe_review():
#     f = open("file_review.pkl","wb")
#     pickle.dump(reviews,f)
#     f.close()
# def store_dataframe_metadata():
#     f = open("file_metadata.pkl","wb")
#     pickle.dump(metadata,f)
#     f.close()


# In[8]:


# store_dataframe_review()
# store_dataframe_metadata()


# In[9]:


def review_get():
    f = open("file_review.pkl","rb")
    return pickle.load(f)
    f.close()

def metadata_get():
    f = open("file_metadata.pkl","rb")
    return pickle.load(f)
    f.close()


# In[10]:


getReviews = review_get()
getMetaData = metadata_get()


# In[11]:


getReviews


# In[12]:


getMetaData


# In[13]:


# getMetaData = getMetaData.drop_duplicates(subset=['asin'])


# In[14]:


# getMetaData


# ### Select 'speaker' as the product

# In[15]:


all_asin_numbers = getMetaData[getMetaData['title'].str.lower().str.contains('speaker', case=False)]['asin'].tolist()


# In[16]:


# all_asin_numbers


# ### No. of Products

# In[17]:


print("Number of Products : ", len(all_asin_numbers))


# In[18]:


getProductRows = getReviews[getReviews['asin'].isin(all_asin_numbers)]


# In[19]:


getProductRows


# ### Number of Available Products

# In[20]:


print("Number of Reviews Available Products : ", getProductRows.shape[0])


# ### Handling of Missing vlaues

# In[21]:


getReviewsAfterMissing = getProductRows.dropna()


# In[22]:


getReviewsAfterMissing


# ### Handling of Duplicate Values

# In[23]:


getReviewsAfterDuplicates = getReviewsAfterMissing.drop_duplicates(subset=['overall', 'vote', 'verified','reviewTime','reviewerID','asin','reviewerName','reviewText','summary','unixReviewTime'])


# In[24]:


getReviewsAfterDuplicates


# ### relevant statistics using EDA

# In[25]:


print("Descriptive Statistics of the speaker : ")
print()
print("Number of Reviews: ",getReviewsAfterDuplicates.shape[0])
print("Average Rating Score: ",getReviewsAfterDuplicates['overall'].mean())
print("Number of Unique Products: ",len(getReviewsAfterDuplicates['asin'].unique()))
print("Number of Good Rating: ", len(getReviewsAfterDuplicates[getReviewsAfterDuplicates['overall']>=3]))
print("Number of Good Rating: ", len(getReviewsAfterDuplicates[getReviewsAfterDuplicates['overall']<3]))
reviews_of_each_rating = getReviewsAfterDuplicates['overall'].value_counts().sort_index(ascending=False)
print()
print("Number of Reviews corresponding to each Rating:")
print(reviews_of_each_rating)


# In[26]:


get_ipython().system('pip3 install unidecode')


# ### Preprocess the Text

# In[27]:


import re
import nltk
# To get stopwords list
nltk.download('stopwords') 
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
import string
from bs4 import BeautifulSoup
def queryProcessing(str_data):

    str_data = BeautifulSoup(str_data,"html.parser").get_text()
    str_data = unidecode(str_data)
    str_data = re.sub('[^a-zA-Z]',' ',str_data)
    str_data = str_data.split()

    # Get english stopwords
    allStopwords = stopwords.words('english')    

    # Add words that are not stopwords
    str_data = [word for word in str_data if not word in set(allStopwords)] 

    lm = WordNetLemmatizer()
    str_data = [lm.lemmatize(token) for token in str_data] 

    
    # join string
    str_temp = ' '.join(str_data)

    str_temp.translate(str.maketrans('', '', string.punctuation))

    str_temp = str_temp.split()

    # Check each word in each string for blank space
    str_data = [i for i in str_temp if i.strip()!='']

    # Join to form string again
    str_data = ' '.join(str_data)
    str_data = str_data.lower()
    return str_data


# In[28]:


getReviewsAfterDuplicates['reviewText'] = getReviewsAfterDuplicates['reviewText'].apply(queryProcessing)


# In[29]:


getReviewsAfterDuplicates


# In[30]:


getCount = getReviewsAfterDuplicates['asin'].value_counts()


# In[31]:


getCount


# ### Get top 20 speaker products

# In[32]:


getTop20Id = getCount.head(24)


# In[33]:


getTop20Id


# ### Get least 20 speaker products

# In[34]:


getLeast20Id = getCount.tail(24)


# In[35]:


getLeast20Id


# In[36]:


top_20 = getMetaData[getMetaData['asin'].isin(getTop20Id.index)]['brand'].value_counts()


# In[37]:


getTop_20 = top_20.index


# In[38]:


print("Top 20 most reviewed brands in speakers: ")
print()
index=1
for i in getTop_20:
    print(f"{index}. {i}")
    index +=1


# In[39]:


least_20 = getMetaData[getMetaData['asin'].isin(getLeast20Id.index)]['brand'].value_counts()


# In[40]:


getLeast_20 = least_20.index


# In[41]:


set_1 = set(getTop_20)
set_2 = set(getLeast_20)
getLeast_20 = set_2 - set_1


# In[42]:


print("Top 20 least reviewed brands in speakers: ")
print()
index=1
for i in getLeast_20:
    print(f"{index}. {i}")
    index +=1


# ### Most positive Reviewed Product

# In[43]:


result_rating_of_5 = getReviewsAfterDuplicates[(getReviewsAfterDuplicates['overall']==5)]


# In[44]:


getResult_of_max_vote_plus_rating_5_review = result_rating_of_5[result_rating_of_5['vote'] == result_rating_of_5['vote'].max()]


# In[45]:


val = getResult_of_max_vote_plus_rating_5_review['asin'].iloc[0]


# In[46]:


print("Most positively reviewed Speaker: ")


# In[47]:


#From Review using 5 rating and high voting
getResult_of_max_vote_plus_rating_5_review


# In[48]:


# From Metadata
getResult_of_max_vote_plus_rating_5_metadata = getMetaData[(getMetaData['asin']==val)]
getResult_of_max_vote_plus_rating_5_metadata


# In[49]:


getProduct_name_1 = getResult_of_max_vote_plus_rating_5_metadata['title']


# In[50]:


for i in getProduct_name_1:
    print(i)


# In[51]:


### If to get Only highly voted product
get_voted_high_only = getReviewsAfterDuplicates[getReviewsAfterDuplicates['vote'] == getReviewsAfterDuplicates['vote'].max()]


# In[52]:


get_voted_high_only


# In[53]:


getResult_of_max_vote_plus_rating_5_metadata


# In[54]:


val = get_voted_high_only['asin'].iloc[0]
getResult_of_highly_vote_only = getMetaData[(getMetaData['asin']==val)]


# In[55]:


getProduct_name_2 = getResult_of_highly_vote_only['title']
for i in getProduct_name_2:
    print(i)


# In[56]:


# To get the positive review by averaging
get_avg_positive_rating = getReviewsAfterDuplicates[getReviewsAfterDuplicates['overall'].isin([3, 4, 5])]
avg_of_good_ratings = get_avg_positive_rating.groupby('asin')['overall'].mean()
getAsin_of_good_reviews = avg_of_good_ratings.idxmax()
# most_positively_reviewed_product_avg_rating = avg_of_good_ratings.max()
getResult_of_max_vote_plus_rating_5_metadata = getMetaData[(getMetaData['asin']==getAsin_of_good_reviews)]


# In[57]:


getResult_of_max_vote_plus_rating_5_metadata.head(1)


# In[58]:


getReviewsAfterDuplicates['year'] = pd.to_datetime(getReviewsAfterDuplicates['unixReviewTime'], unit='s').dt.year


# In[59]:


getReviewsAfterDuplicates


# ### count of ratings for the product over 5 consecutive years.

# In[72]:


getYears = getReviewsAfterDuplicates.groupby('year')['overall'].count()
get_5_Years = getYears.loc[2010:2015]
print("Ratings for the product over 5 consecutive years: ")
print(get_5_Years)


# ### Word Cloud

# In[73]:


goodReviewList = []
badReviewList = []
for index,row in getReviewsAfterDuplicates.iterrows():
    if(row['overall']>=3):
        goodReviewList.append(row['reviewText'])
    else:
        badReviewList.append(row['reviewText'])


# In[74]:


goodReviews = ' '.join(goodReviewList)
badReviews = ' '.join(badReviewList)


# In[75]:


goodReviews


# In[76]:


badReviews


# In[77]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
allStopwords = stopwords.words('english') 
wc1 = WordCloud(max_font_size=50, max_words=100, background_color="white",stopwords=allStopwords).generate(goodReviews)
plt.figure()
plt.imshow(wc1, interpolation="bilinear")
plt.title('Good Ratings WordCloud')
plt.axis("off")
plt.show()


# In[79]:


allStopwords = stopwords.words('english') 
wc2 = WordCloud(max_font_size=50, max_words=100, background_color="white",stopwords=allStopwords).generate(badReviews)
plt.figure()
plt.imshow(wc2, interpolation="bilinear")
plt.title('Bad Ratings WordCloud')
plt.axis("off")
plt.show()


# wc2 = WordCloud(max_font_size=50, max_words=100, background_color="white",stopwords=allStopwords).generate(badReviews)
# plt.figure()
# plt.imshow(wc2, interpolation="bilinear")
# plt.title('Bad Ratings WordCloud')
# plt.axis("off")
# plt.show()

# ### commonly used words for positive and negative reviews

# In[80]:


goodReview_count = wc1.words_
badReview_count = wc2.words_


# In[81]:


topGoodReviews = sorted(goodReview_count.items(), key=lambda x: x[1], reverse=True)[:10]
topBadReviews = sorted(badReview_count.items(), key=lambda x: x[1], reverse=True)[:10]


# In[82]:


print("Top 10 Good review words: ")
print()
for i in topGoodReviews:
    print(i[0])


# In[83]:


print("Top 10 Bad review words: ")
print()
for i in topBadReviews:
    print(i[0])


# ### Plot a pie chart for Distribution of Ratings vs. the No. of Reviews.

# In[84]:


getReviewClassify = getReviewsAfterDuplicates['overall'].value_counts()


# In[85]:


getReviewClassify


# In[86]:


plt.figure(figsize=(6, 6))
plt.pie(getReviewClassify, labels=getReviewClassify.index, autopct='%1.1f%%', startangle=180)
plt.title('Pie chart plot for Distribution of Ratings vs. the No. of Reviews.')
plt.axis('equal')
plt.show()


# ### Report in which year the product got maximum reviews.

# In[87]:


maxReviewYear = getReviewsAfterDuplicates['year'].value_counts().index
print("year in which speaker got maximum reviews: ",maxReviewYear[0])


# ### year having the highest number of Customers

# In[88]:


getHighestReviewYear = getReviewsAfterDuplicates.groupby('year')['reviewerID'].nunique()


# In[89]:


getHighestReviewYear


# In[90]:


print("Year has the highest number of Customers: ", getHighestReviewYear.idxmax())


# In[91]:


get_ipython().system('pip3 install gensim')


# ### Word2Vec feature engineering

# In[92]:


getReviewText = getReviewsAfterDuplicates['reviewText']


# In[93]:


from gensim.models import Word2Vec
word2vec_model = Word2Vec(sentences=getReviewText, vector_size=100, window=5, min_count=1, workers=4)


# In[94]:


X = getReviewText.apply(lambda review: [word2vec_model.wv[word] for word in review if word in word2vec_model.wv])


# ### Rating Class is divided into three categories

# In[95]:


getReviewsAfterDuplicates['rating_class'] = pd.cut(getReviewsAfterDuplicates['overall'], bins=[0, 2.99, 3, 5], labels=['Bad', 'Average', 'Good'])
Y = getReviewsAfterDuplicates['rating_class']


# ### Train-test split

# In[96]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
import numpy as np
x_train = np.vstack([np.mean(vectors, axis=0) for vectors in x_train])
x_test = np.vstack([np.mean(vectors, axis=0) for vectors in x_test])


# ### Machine Learning Model

# In[97]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# In[98]:


allmodels = {
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassfier": RandomForestClassifier(),
    "SVC": SVC(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier":DecisionTreeClassifier(),
    "GaussianNB": GaussianNB()
}

dictionaryModelResults = {}

for classifier, model in allmodels.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    results = classification_report(y_test, y_pred, output_dict=True)
    dictionaryModelResults[classifier] = results

for Classifier, results in dictionaryModelResults.items():
    print(f"Classification Report for {Classifier}:")
    print()
    for Reviewtype, parameters in results.items():
        if Reviewtype in ['Good', 'Average', 'Bad']:
            print(f"Review Type: {Reviewtype}")
            print(f"Precision: {parameters['precision']}")
            print(f"Recall: {parameters['recall']}")
            print(f"F1-score: {parameters['f1-score']}")
            print(f"Support: {parameters['support']}")
            print()


# ### user-item rating matrix

# In[99]:


getParameters = ['reviewerID', 'asin', 'overall']
review_data = getReviewsAfterDuplicates[getParameters]
user_item_matrix = review_data.pivot_table(index='reviewerID', columns='asin', values='overall', aggfunc='mean')
pd.set_option('display.max_rows', None)


# ### min-max scaling

# In[100]:


from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
normalized_values = mm.fit_transform(user_item_matrix)


# In[101]:


normalized_values_dataframe = pd.DataFrame(normalized_values, index=user_item_matrix.index, columns=user_item_matrix.columns)


# ### user-user recommender system

# In[102]:


from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error

vector = [10, 20, 30, 40, 50]

error1 = []
fold = KFold(n_splits=5)

getNormalized_values_dataframe = normalized_values_dataframe.fillna(normalized_values_dataframe.mean())

for val in vector:
    getErrors = []
    for id1, id2 in fold.split(getNormalized_values_dataframe):
        output = []
        getId2 = getNormalized_values_dataframe.iloc[id2]
        getId1 = getNormalized_values_dataframe.iloc[id1]
        getSimilar = cosine_similarity(getId1)
        for num, (_, line) in enumerate(getId2.iterrows()):
            # sm = getSimilar[num]
            getIndex = getSimilar[num].argsort()[-(val+1):-1]
            # getRating = getId1.iloc[getIndex]
            # curr = getRating.mean(axis=0)
            curr = getId1.iloc[getIndex].mean(axis=0)
            output.append(curr)
        output = [ptr for j in output for ptr in j]
        original_val = getId2.values.flatten()
        e = mean_absolute_error(original_val, output)
        getErrors.append(e)
    # getAvg = sum(getErrors) / 5
    error1.append(sum(getErrors) / 5)
print("MAE (Mean Absolute Error) for user-user recommender system: ")
for index, val in enumerate(vector):
    print(f"For {val} =  {error1[index]}")


# ### item-item recommender system

# In[103]:


import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vector = [10, 20, 30, 40, 50]

error2 = []
fold = KFold(n_splits=5)

getNormalized_values_dataframe = normalized_values_dataframe.fillna(normalized_values_dataframe.mean())

for val in vector:
    getErrors = []
    for id1, id2 in fold.split(getNormalized_values_dataframe):
        output = []
        getId2 = getNormalized_values_dataframe.iloc[id2]
        getId1 = getNormalized_values_dataframe.iloc[id1]
        getSimilar = cosine_similarity(getId1.T)

        
        for num, (_, line) in enumerate(getId2.iterrows()):
            sm = getSimilar[num]
            getIndex = sm.argsort()[-(val+1):-1]
            getRating = getId1.T.iloc[getIndex]
            curr = (getRating * sm[getIndex][:, None]).sum(axis=1) / sm[getIndex].sum()
            output.append(curr)

        output = [ptr for j in output for ptr in j]

        original_val = getId2.values.flatten()

        getLen = min(len(output), len(original_val))
        output = output[:getLen]
        original_val = original_val[:getLen]

        output = np.nan_to_num(output)

        e = mean_absolute_error(original_val, output)
        getErrors.append(e)
    error2.append(sum(getErrors) / 5)
    
print("MAE (Mean Absolute Error) for item-item recommender system: ")
for index, val in enumerate(vector):
    print(f"For {val} =  {error2[index]}")


# ### MAE against K plot

# In[104]:


vector = [10, 20, 30, 40, 50]

plt.figure(figsize=(12, 8))
plt.plot(vector, error1, marker='o', linestyle='-')
plt.xlabel('K')
plt.ylabel('MAE')
plt.title('MAE vs K for user-user recommendation system')
plt.xticks(vector)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(vector, error2, marker='o', linestyle='-')
plt.xlabel('K')
plt.ylabel('MAE')
plt.title('MAE vs K for item-item recommendation system')
plt.xticks(vector)
plt.grid(True)
plt.tight_layout()
plt.show()


# ### TOP 10 products by User Sum Ratings.

# In[105]:


top_user_sum_ratings = getReviewsAfterDuplicates.groupby('asin')['overall'].sum()
getProducts = top_user_sum_ratings.sort_values(ascending=False).head(10).index


# In[106]:


top_10_results_user_sum_ratings = getMetaData[getMetaData['asin'].isin(getProducts)]['title']


# In[107]:


index=1
for i in top_10_results_user_sum_ratings:
    print(f"{index}. {i}")
    index +=1

