#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd


# In[5]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[6]:


# movies.head(1)


# In[7]:


# movies.shape


# In[8]:


# credits.shape


# In[9]:


# merging the tables i.e taking title 
movies = movies.merge(credits, on='title')


# In[10]:


# nowthe shape will be 23 not 24 because on colummn is merged
# movies.shape


# In[11]:


# movies.info()


# In[12]:


# now filtering out data 
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[13]:


# movies.head(2)


# In[16]:


# now more filteration to each column
# movies.isnull().sum()


# In[15]:


#overview as 3 null rows , so filtering it 
#dropna(): This method is used in pandas to remove missing values
#inplace=True modifies the original DataFrame (movies) directly, instead of returning a modified copy.
movies.dropna(inplace=True)


# In[17]:


# movies.duplicated().sum()


# In[26]:


# movies.iloc[0].genres


# In[28]:


import ast 


# In[31]:


# we just want thsi  Action,Adventure,Fantasy
#creating a function to clean genres 
def convert(obj):
    L = []
    for i in ast.literal_eval(obj): # ast_literal_eval works on object as tuple,dictionary,string
        L.append(i['name'])
    return L


# In[33]:


movies['genres'] = movies['genres'].apply(convert)


# In[34]:


# movies.head()


# In[36]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[37]:


# movies.head()


# In[40]:


# movies['cast'][0]


# In[45]:


#now for cast Column we only need first 3 actors
def convert_three(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if(counter >= 3):
            break
        else:
            L.append(i['name'])
            counter +=1
    return L


# In[48]:


movies['cast'] = movies['cast'].apply(convert_three)


# In[49]:


# movies.head()


# In[50]:


# movies['crew'][0]


# In[56]:


# now To extract Director only, which is only one for each movie
def extract_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L


# In[59]:


movies['crew'] = movies['crew'].apply(extract_director)


# In[60]:


# movies.head()


# In[64]:


#now for the overview
movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[65]:


# movies.head(2)


# In[69]:


#now removing any " " space between 
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[70]:


# movies.head(2)


# In[71]:


#creating new column to merge five columns into one
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
# movies.head()


# In[76]:


#creating new table of only usefull columns
new_df_movies = movies[['movie_id','title','tags']]
# new_df_movies


# In[77]:


# converting tags list to string
new_df_movies['tags'] = new_df_movies['tags'].apply(lambda x:" ".join(x))


# In[85]:


# new_df_movies['tags'][0]


# In[86]:


# new_df_movies.head()


# In[87]:


#coneverting to lowercase
new_df_movies['tags'].apply(lambda x:x.lower())
# new_df_movies.head()


# In[98]:


#to remove similar words using nltk
# get_ipython().system('pip install nltk')


# In[99]:


# using nltk to remove duplicate words or adjectives etc
import nltk


# In[100]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[101]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[102]:


#only checking
# stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')


# In[104]:


new_df_movies['tags'] = new_df_movies['tags'].apply(stem)


# In[105]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[108]:


vectors = cv.fit_transform(new_df_movies['tags']).toarray()


# In[109]:


# vectors[0]


# In[114]:


# len(cv.get_feature_names_out())
#np.set_printoptions(threshold=np.inf)
#print(cv.get_feature_names_out())


# In[115]:


from sklearn.metrics.pairwise import cosine_similarity


# In[122]:


similarity = cosine_similarity(vectors)


# In[123]:


# similarity[0]


# In[121]:


#enumerate() also returns the th_number of movies 
#list wraps string into list so can be sorted
#reverse sorts allows to sort in descending order i.e the higest value first
#key=lambda x:x[1] tells to sort on second column
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[136]:


def movie_recommender(movie):
    movie_index = new_df_movies[new_df_movies['title'] == movie].index[0]
    distances = similarity[movie_index] # here similarties are not sorted 
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6] # sorts similarities in reverse and gives only five 

    movie_names=[]
    for i in movies_list:
        print(new_df_movies.iloc[i[0]].title)
        movie_names.append(new_df_movies.iloc[i[0]].title)

    return movie_names

# In[1]:


#movie_recommender("Fifty Shades of Grey")






