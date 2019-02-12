import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor as GBT
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import gc

HIDDEN_STATE_SIZE = 6
MAX_ITERATION = 10000

temp_X = np.load('Xtrain.npz')['X']
y = np.load('ytrain_raw.npz')['y']


X = np.zeros((temp_X.shape[0], temp_X.shape[1]+HIDDEN_STATE_SIZE))
X[:,:-HIDDEN_STATE_SIZE] = temp_X

Xtrain = X[:len(X)//2]
ytrain = y[:len(y)//2]

Xval = X[len(X)//2:]
yval = y[len(y)//2:]


temp_Xtest = np.load('Xtest.npz')['X']
ytest = np.load('ytest_raw.npz')['y']

Xtest = np.zeros((temp_Xtest.shape[0], temp_Xtest.shape[1]+HIDDEN_STATE_SIZE))
Xtest[:,:-HIDDEN_STATE_SIZE] = temp_Xtest


print("x shape", Xtrain.shape, "y shape", ytrain.shape )

clf = GBT(n_estimators = 10, warm_start = True)
print(clf)

clf.fit(Xtrain, ytrain)
ytest_pred = clf.predict(Xtest)


for iter in range(MAX_ITERATION):
	print("ITERATION #", iter)

	clf.n_estimators += 1

	ytrain_pred = np.array([-1000 for k in range(HIDDEN_STATE_SIZE)])
	
	ytrain_pred = np.append(ytrain_pred, clf.predict(Xtrain))

	ytrain_pred = np.append(ytrain_pred, np.array([-1000 for k in range(HIDDEN_STATE_SIZE)]))
	
	for i in range(len(ytrain_pred)- 2*HIDDEN_STATE_SIZE):
		window = np.array([ytrain_pred[i: i+HIDDEN_STATE_SIZE]])
		Xtrain[i,-HIDDEN_STATE_SIZE:] = window
	
	# train again with new features
	clf.fit(Xtrain, ytrain)

	print("Training MAE = ", mae(ytrain, ytrain_pred[HIDDEN_STATE_SIZE: -HIDDEN_STATE_SIZE]))




	# Testing to be done here
	ytest_pred = np.array([-1000 for k in range(HIDDEN_STATE_SIZE)])

	ytest_pred = np.append(ytest_pred, clf.predict(Xtest))

	ytest_pred = np.append(ytest_pred, np.array([-1000 for k in range(HIDDEN_STATE_SIZE)]))


	for i in range(len(ytest_pred)- 2*HIDDEN_STATE_SIZE):
		window = np.array([ytest_pred[i: i+HIDDEN_STATE_SIZE]])
		Xtest[i,-HIDDEN_STATE_SIZE:] = window


	print("Testing MAE = ", mae(ytest, ytest_pred[HIDDEN_STATE_SIZE: -HIDDEN_STATE_SIZE]))
