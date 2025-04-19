# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 08:56:34 2025

@author: NTCuong
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

utility_matrix = np.array([[5, 5, 0, 0, 1, '?', '?'],
                           [5, '?', '?', 0, '?', '?', '?'],
                           ['?', 4, 1, '?', '?', 1, 2],
                           [1, 1, 4, 4, 4, '?', 4],
                           [1, 0, 5, '?', '?', '?', 5]])

utility_matrix_0 = utility_matrix.copy()
n_user = utility_matrix.shape[1]
mean_users = np.zeros(n_user)
for user_index in np.arange(n_user):
    ratings = utility_matrix_0[:, user_index]
    ids = np.where(ratings == '?')
    ratings[ids] = 0

print(utility_matrix_0, type(utility_matrix_0))
utility_matrix_f = utility_matrix_0.astype(float)
print(utility_matrix_f)


def normalize_Y(array):
    array_0 = array.copy()
    n_user = array_0.shape[1]
    mean_users = np.zeros(n_user)
    for user_index in np.arange(n_user):
        ratings = array_0[:, user_index]
        ids = np.where(ratings == '?')
        ratings[ids] = 0

    array_f = array_0.astype(float)
    for user_index in np.arange(n_user):
        ratings = array_f[:, user_index]
        ratings_0 = array[:, user_index]
        m = np.mean(ratings)
        mean_users[user_index] = m
        ids = np.where(ratings_0 != '?')
        array_f[ids, user_index] = ratings[ids] - mean_users[user_index]
    return mean_users, array_f

mean_users, utility_matrix_bar = normalize_Y(utility_matrix)
print(mean_users, utility_matrix_bar)


simi_user_matrix = cosine_similarity(utility_matrix_bar.T, utility_matrix_bar.T)
print(simi_user_matrix)









