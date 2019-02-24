import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor as GBT
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
# from catboost import CatBoostRegressor as CBR
import gc

HIDDEN_STATE_SIZE = 10
MAX_ITERATION = 1000
WINDOW_SIZE = 21

temp_X = np.load('Xtrain' + str(WINDOW_SIZE)+ '.npz')['X']
y = np.load('ytrain_raw.npz')['y']

# X = np.zeros((temp_X.shape[0], temp_X.shape[1]+HIDDEN_STATE_SIZE))
# X[:,:-HIDDEN_STATE_SIZE] = temp_X

X = temp_X

Xtrain = X
ytrain = np.sqrt(y)

# Xval = X[len(X)//2:]
# yval = y[len(y)//2:]


temp_Xtest = np.load('Xtest' + str(WINDOW_SIZE)+ '.npz')['X']
ytest = np.load('ytest_raw.npz')['y']
# ytest = np.sqrt(ytest)
# Xtest = np.zeros((temp_Xtest.shape[0], temp_Xtest.shape[1]+HIDDEN_STATE_SIZE))
# Xtest[:,:-HIDDEN_STATE_SIZE] = temp_Xtest
Xtest = temp_Xtest

print("x shape", Xtrain.shape, "y shape", ytrain.shape )

# clf = GBT(n_estimators = 1000, max_depth = 3, warm_start = True)
clf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)

# clf = CBR(max_depth = 16)
print(clf)

clf.fit(Xtrain, ytrain)
# ytest_pred = clf.predict(Xtest)


# print("ITERATION #", iter)

#clf.n_estimators += 1

# ytrain_pred = np.array([-1000 for k in range(HIDDEN_STATE_SIZE)])

# ytrain_pred = np.append(ytrain_pred, clf.predict(Xtrain))
ytrain_pred = clf.predict(Xtrain)
# ytrain_pred = np.append(ytrain_pred, np.array([-1000 for k in range(HIDDEN_STATE_SIZE)]))

# for i in range(len(ytrain_pred)- 2*HIDDEN_STATE_SIZE):
# 	window = np.array([ytrain_pred[i: i+HIDDEN_STATE_SIZE]])
# 	Xtrain[i,-HIDDEN_STATE_SIZE:] = window

# train again with new features
# clf.fit(Xtrain, ytrain)
ytrain_pred = np.square(ytrain_pred)
ytrain = np.square(ytrain)
print("Training MAE = ", mae(ytrain, ytrain_pred))




# Testing to be done here
# ytest_pred = np.array([-1000 for k in range(HIDDEN_STATE_SIZE)])

# ytest_pred = np.append(ytest_pred, clf.predict(Xtest))
ytest_pred = clf.predict(Xtest)
# ytest_pred = np.append(ytest_pred, np.array([-1000 for k in range(HIDDEN_STATE_SIZE)]))


# for i in range(len(ytest_pred)- 2*HIDDEN_STATE_SIZE):
# 	window = np.array([ytest_pred[i: i+HIDDEN_STATE_SIZE]])
# 	Xtest[i,-HIDDEN_STATE_SIZE:] = window

ytest_pred = np.square(ytest_pred)
# ytest = np.square(ytest)
print("Testing MAE = ", mae(ytest, ytest_pred))
