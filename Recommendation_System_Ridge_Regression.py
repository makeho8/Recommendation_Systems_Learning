# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 09:50:22 2025

@author: Admin
Recommendation Systems
"""
import pandas as pd 

# Đọc file user:
u_cols =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names = u_cols,
 encoding='latin-1')

n_users = users.shape[0]
print('Số lượng users:', n_users)
users.head() # xem 5 user đầu tiên

# Đọc file ratings:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.to_numpy()
rate_test = ratings_test.to_numpy()

print('Number of training rates:', rate_train.shape)
print('Number of test rates:', rate_test.shape)

# load items information to items variable
#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')

n_items = items.shape[0]
print('Number of items:', n_items)

X0 = items.to_numpy()
X_train_counts = X0[:, -19:] # 19 gia tri nhi phan cuoi moi hang

# feature vector via TFIDF
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf = True, norm ='l2')
tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray()

# Tim item ma user da rated
import numpy as np
def get_items_rated_by_user(rate_matrix, user_ID):
    """
    in each line of rate_matrix, we have infor: [user_id, item_id, rating (scores), time_stamp]
    we care about the first three values
    return (item_indices, scores) rated by user user_id
    """
    y = rate_matrix[:,0] # all users
    # item indices rated by user_index
    # we need to +1 to user_index since in the rate_matrix, id starts from 1 
    # while index in python starts from 0
    #user_ID = index + 1
    indices = np.where(y == user_ID)[0] # tim cac indices (vi du: [3, 5, 7, ...]) co element là user_ID
    item_IDs = rate_matrix[indices, 1]
    item_indices = item_IDs -1 # index starts from 0 
    scores = rate_matrix[indices, 2]
    return (item_indices, scores, indices)

# Tim mo hinh cho moi user
from sklearn.linear_model import Ridge
from sklearn import linear_model

d = tfidf.shape[1] # data dimension
W = np.zeros((d, n_users))
b = np.zeros((1, n_users))

for user_index in range(n_users):    
    user_ID = user_index + 1
    item_indices, scores, indices = get_items_rated_by_user(rate_train, user_ID)
    clf = Ridge(alpha=0.01, fit_intercept  = True)
    Xhat = tfidf[item_indices, :]
    
    clf.fit(Xhat, scores) 
    W[:, user_index] = clf.coef_
    b[0, user_index] = clf.intercept_
    
# predicted scores
Yhat = tfidf.dot(W) + b

# Vi du voi user có ID la 1
n = 1
np.set_printoptions(precision=2) # 2 digits after . 
item_indices, scores, indices = get_items_rated_by_user(rate_test, n)
Yhat[item_indices, n]
print('Rated items indices:', item_indices)
print('True ratings     :', scores)
print('Predicted ratings:', Yhat[item_indices, n])
print("indices for user_ID_n:", indices)

# Danh gia mo hinh
def evaluate(Yhat, rates, W, b):
    se = 0
    cnt = 0
    for n in range(n_users):
        item_ids, scores_truth, indices = get_items_rated_by_user(rates, n)
        scores_pred = Yhat[item_ids, n]
        e = scores_truth - scores_pred 
        se += (e*e).sum(axis = 0)
        cnt += e.size 
    return np.sqrt(se/cnt)

print ('RMSE for training:', evaluate(Yhat, rate_train, W, b))
print ('RMSE for test    :', evaluate(Yhat, rate_test, W, b))