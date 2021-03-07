# list of functions to be called up within the collab rating main files
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def generate_random_reviews(n,r,f):
    # n = no. of films/shows
    # r = no. of reviewers
    # f = frac of reviews in total

    # review ratings to be drawn from a normal dist
    rating_mean = 5
    rating_std = 1

    ratings = np.zeros((n,r)) # initialise with zeros
    rng = np.random.default_rng()

    for i in range(r):
        inds = rng.choice(n,np.int(f*n))
        ratings[inds,i] = np.random.normal(rating_mean,rating_std,len(inds))

    return ratings

def calc_similarity(ratings,kind='user'):
    # calculate cosine distance pairwise between elements of the ratings matrix

    if kind == 'user':
        sim = np.dot(ratings,ratings.T) + 1e-9
    elif kind == 'item':
        sim = np.dot(ratings.T, ratings) + 1e-9

    norm_sq = np.sqrt(np.diagonal(sim))[np.newaxis,:]
    return sim/norm_sq/norm_sq.T

def calc_bias(ratings,type='user'):
    # calculate bias (average rating scores) of raters over scored (non-zero) entries
    if type == 'user':
        bias = np.zeros([ratings.shape[0],1])
        unbiased_ratings = np.zeros(ratings.shape)

        for i in range(ratings.shape[0]):
            rating_i = ratings[i, :]
            nonzero_entries = np.nonzero(rating_i)
            bias[i, 0] = np.average(rating_i[nonzero_entries])
            rating_i[nonzero_entries] = rating_i[nonzero_entries] - bias[i, 0]
            unbiased_ratings[i,:] = rating_i

    elif type == 'item':
        bias = np.zeros([1,ratings.shape[1]])
        unbiased_ratings = np.zeros(ratings.shape)

        for i in range(ratings.shape[1]):
            rating_i = ratings[:, i]
            nonzero_entries = np.nonzero(rating_i)
            bias[0, i] = np.average(rating_i[nonzero_entries])
            rating_i[nonzero_entries] = rating_i[nonzero_entries] - bias[0, i]
            unbiased_ratings[:,i] = rating_i

    # return bias and bias-subtracted ratings matrix
    return bias, unbiased_ratings

def predict_user_based(ratings,user_similarity,user_bias):
    # unbiased user-similarity based predictor for ratings across all users

    preds = np.dot(user_similarity,ratings-user_bias) # unbiased prediction
    norm_preds = (preds / np.sum(user_similarity,axis=1)[:,np.newaxis]) + user_bias

    return norm_preds

def predict_item_based(ratings,item_similarity,item_bias):
    # unbiased user-similarity based predictor for ratings across all users

    preds = np.dot(ratings-item_bias,item_similarity) # unbiased prediction
    norm_preds = (preds / np.sum(item_similarity,axis=0)[np.newaxis,:]) + item_bias

    return norm_preds

def rms_pred_error(test_ratings,preds_user,preds_item):
    # RMS error of item and user similaity based rating predictions against a test sub-set

    valid_test_ratings = test_ratings[:,:-1]
    test_indices = test_ratings[:,-1]
    test_indices = test_indices.astype(int)
    preds_user_test = preds_user[test_indices,:]
    preds_item_test = preds_item[test_indices,:]

    # non-zero, i.e. scored ratings in test dataset
    scored_indices = np.nonzero(valid_test_ratings)
    user_pred_rmse = mean_squared_error(valid_test_ratings[scored_indices], preds_user_test[scored_indices],squared=False)
    try:
        item_pred_rmse = mean_squared_error(valid_test_ratings[scored_indices], preds_item_test[scored_indices],squared=False)
    except ValueError:
        print('The ratings data set is too sparse for the item similarity to be calculated')
        item_pred_rmse = 0

    return user_pred_rmse, item_pred_rmse