# Script to create and analyse a randomised, simulated ratings dataset for the NetPrime Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
import seaborn as sns
import collab_rating_funcs as col_func

data = pd.read_excel(r'C:\Users\slaye\Desktop\Python\Collab Rating\Indian_NetPrime_DataSet_edited.xls')

# create r simulated reviewers
n = len(data)
r = 100
# base = data[['Sid']].fillna(value=0).to_numpy()
random_ratings = col_func.generate_random_reviews(n,r,0.2)
# ratings = np.concatenate([base,random_ratings],axis=1).T
ratings = random_ratings.T

# Similarity analysis - Method 1 - simple user and item based
user_similarity = col_func.calc_similarity(ratings,kind='user')
item_similarity = col_func.calc_similarity(ratings,kind='item')

# predict missing entries in dataset
u_bias, unbiased_u_ratings = col_func.calc_bias(ratings.copy(),type='user')
i_bias, _ = col_func.calc_bias(ratings.copy(),type='item')
preds_user = col_func.predict_user_based(ratings,user_similarity,u_bias)
preds_item = col_func.predict_item_based(ratings,item_similarity,i_bias)

# Evaluate accuracy of predictions against a randomised test sub-dataset
indexed_ratings = np.concatenate((ratings,np.arange(r)[:,np.newaxis]),axis=1)
_, test_ratings = train_test_split(indexed_ratings, test_size=0.25, random_state=1)
user_pred_rmse, item_pred_rmse = col_func.rms_pred_error(test_ratings,preds_user,preds_item)

print('User-based CF RMSE for UOI: ' + str(user_pred_rmse))
print('Item-based CF RMSE for UOI: ' + str(item_pred_rmse))

# Similarity analysis - Method 2 - Matrix Factorization using SVD for user based collab filtering
from scipy.sparse.linalg import svds

# get SVD components from train matrix. Choose k arbitrarily
indexed_unbiased_ratings = np.concatenate((unbiased_u_ratings,np.arange(r)[:,np.newaxis]),axis=1)
_, test_unbiased_ratings = train_test_split(indexed_unbiased_ratings, test_size=0.25)

u, s, vt = svds(unbiased_u_ratings, k = 75)
s_diag_matrix = np.diag(s)
ratings_pred_svd = np.dot(np.dot(u, s_diag_matrix), vt) + u_bias
rmse_svd, _ = col_func.rms_pred_error(ratings,ratings_pred_svd,np.zeros(ratings_pred_svd.shape))
print('User-based CF RMSE SVD: ' + str(rmse_svd))

# Distribution Plots for User of Interest
uoi = 0
ratings_uoi = ratings[uoi,:]
preds_user_uoi = preds_user[uoi,:]
preds_item_uoi = preds_item[uoi,:]
preds_svd_uoi = ratings_pred_svd[uoi,:]

plt.figure()
sns.distplot(ratings_uoi[ratings_uoi.nonzero()],kde=False,bins=50,norm_hist=True)
sns.distplot(preds_user_uoi[preds_user_uoi.nonzero()],kde=False,bins=50,norm_hist=True)
sns.distplot(preds_item_uoi[preds_item_uoi.nonzero()],kde=False,bins=50,norm_hist=True)
sns.distplot(preds_svd_uoi,kde=False,bins=50,norm_hist=True)
plt.legend(['user ratings','user-similarity predictions','item-similarity predictions','svd predictions'])
plt.title('predicted ratings distribution for UOI')

# Matrix Factorization shows much lower RMS errors hence will be used for recommendations
uoi_entries = np.nonzero(ratings[uoi,:])
preds_svd_uoi[uoi_entries] = 0 # set already rated items to 0 so it wont show up in top 5 sorting

top5_towatch_user = data['Title'][np.argsort(-1*preds_user_uoi)[:5]]
top5_towatch_item = data['Title'][np.argsort(-1*preds_item_uoi)[:5]]
top5_towatch_svd = data['Title'][np.argsort(-1*preds_svd_uoi)[:5]]
