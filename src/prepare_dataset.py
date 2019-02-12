import os
import numpy as np
import random
import gc
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae


random.seed(42)

aa = dict(
    A = 20,
    R = 1,
    N = 2,
    D = 3,
    C = 4,
    Q = 5,
    E = 6,
    G = 7,
    H = 8,
    I = 9,
    L = 10,
    K = 11,
    M = 12,
    F = 13,
    P = 14,
    S = 15,
    T = 16,
    W = 17,
    Y = 18,
    V = 19
)

def prepareDataSet(files):
    SAMPLE_INTERVAL = 200
    MAX_FILE_COUNT = 10000000000
    FEATURE_DIMENSION = 21

    X = np.empty((0,FEATURE_DIMENSION), int)
    y = np.empty((0,1), float)


    file_count = 0
    for file in files:
        file_count += 1
        if(file_count % SAMPLE_INTERVAL == 0):
            print(file)
        if(file_count >= MAX_FILE_COUNT):
            break

        input = np.genfromtxt("../data/"+file, dtype=None, encoding=None)

        seq = [0 for i in range(FEATURE_DIMENSION//2)]
        for(serial, acid, asa) in input[2:]:
            if(acid in aa):
                seq += [aa[acid]]
                y = np.append(y, asa)
        
        # y = (y-y.mean())/ y.std()

        seq += [0 for i in range(FEATURE_DIMENSION//2)]

        for i in range(len(seq) - FEATURE_DIMENSION+1):
            window = np.array([seq[i: i+FEATURE_DIMENSION]])
            # print(window.shape)
            X = np.append(X, window, axis = 0)

    print("x shape", X.shape, "y shape", y.shape )
    print(X[len(X)-1])
    print(X[0])
    return X , y


if __name__ == "__main__":
    all_files = os.listdir("../data/")
    random.shuffle(all_files)
    train_files = all_files[:int(0.8*len(all_files))]
    test_files = all_files[int(0.8*len(all_files)):]


    Xtrain, ytrain = prepareDataSet(train_files)

    # clf = RandomForestRegressor(n_estimators=100, criterion='mae', n_jobs=4, max_depth=3)
    # clf.fit(Xtrain, ytrain)
    # del Xtrain, ytrain
    # gc.collect()

    Xtest, ytest = prepareDataSet(test_files)
    # ytest_pred = clf.predict(Xtest)
    # print(mae(ytest, ytest_pred))

    np.savez_compressed('Xtrain21.npz', X=Xtrain)
    np.savez_compressed('Xtest21.npz', X=Xtest)
    np.savez_compressed('ytrain21.npz', y=ytrain)
    np.savez_compressed('ytest21.npz', y=ytest)