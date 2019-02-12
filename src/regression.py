import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
import gc


Xtrain = np.load('Xtrain21.npz')['X']
ytrain = np.load('ytrain21.npz')['y']
print("x shape", Xtrain.shape, "y shape", ytrain.shape )

clf = RandomForestRegressor(n_estimators=200, max_depth=3)
clf.fit(Xtrain, ytrain)

del Xtrain, ytrain
gc.collect()

Xtest = np.load('Xtest21.npz')['X']
ytest = np.load('ytest21.npz')['y']
ytest_pred = clf.predict(Xtest)
print(mae(ytest, ytest_pred))