import math

import numpy as np

from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn import svm

# from sklearn.cross_validation import train_test_split #cross_val_score
from sklearn.cross_validation import KFold

# For model I/O
from sklearn.externals import joblib

from root_numpy import root2rec #root2array

trainVars = []

def createGBRT(learning_rate=0.02, max_depth=5, n_estimators=500, subSample=0.5):
    clf = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=1, loss='lad', verbose=1, subsample=subSample, max_features=0.5, min_samples_leaf=1, min_samples_split=1)#'auto', loss=huber
    return clf

def createSVR(C=0.1):
    # clf = svm.SVR(C=10000.0, verbose=2, random_state=1, epsilon=0.001)
    clf = svm.SVR(C=C, verbose=2, random_state=1, epsilon=0.5, cache_size=1000., degree=3)
    return clf

def createNuSVR():
    # clf = svm.SVR(C=10000.0, verbose=2, random_state=1, epsilon=0.001)
    clf = svm.NuSVR(nu=0.3, C=100000.0, verbose=0, random_state=1, degree=3, gamma=0.)
    return clf

def createAdaSVR():
    clf = AdaBoostRegressor(base_estimator=svm.SVR(C=10000.0, verbose=0, random_state=1, epsilon=0.005, degree=3), loss='square', learning_rate=0.3, n_estimators=5, random_state=1)
    return clf

def createBaggingSVR():
    # clf = BaggingRegressor(base_estimator=svm.SVR(C=100000.0, verbose=0, random_state=1, epsilon=0.1, degree=3, kernel='rbf'),  max_features=0.1, n_estimators=100, oob_score=True, random_state=1, verbose=1, n_jobs=2, max_samples=0.8)
    clf = BaggingRegressor(base_estimator=svm.SVR(C=1.0, verbose=0, random_state=1, epsilon=1.0, degree=3, kernel='rbf'),  max_features=0.3, n_estimators=100, oob_score=True, random_state=1, verbose=1, n_jobs=2, max_samples=0.8, bootstrap_features=False)
    return clf

def createBaggingRidge():
    clf = BaggingRegressor(base_estimator=svm.NuSVR(nu=0.3, C=10000.0, verbose=0, random_state=1, degree=3, gamma=0.),  max_features=0.1, n_estimators=100, oob_score=True, random_state=1, verbose=1, n_jobs=1, max_samples=0.8, bootstrap_features=False)
    return clf


def train(clf, training_data, targets, weights, nCrossVal=4, var_lists=[]):
    
    print 'Using classifier'
    print clf

    kf = KFold(len(training_data), nCrossVal, shuffle=True, random_state=1) # 4-fold cross-validation

    print 'Cross-validation:', nCrossVal, 'folds'

    targetSum2 = 0.
    targetSum = 0.
    
    for trainIndices, testIndices in kf:
        d_train = training_data[trainIndices]
        d_test = training_data[testIndices]

        t_train = targets[trainIndices]
        t_test = targets[testIndices]

        w_train = weights[trainIndices] if weights else None
        w_test = weights[testIndices] if weights else None

        clf.fit(d_train, t_train, sample_weight=w_train)
        test_scores = clf.predict(d_test)

        if weights:
            sum2 = sum(w*(s-t)**2 for s, t, w in zip(test_scores, t_test, w_test))
            suml = sum(w*abs(s-t) for s, t, w in zip(test_scores, t_test, w_test))
        else:
            sum2 = sum((s-t)**2 for s, t in zip(test_scores, t_test))
            suml = sum(abs(s-t) for s, t in zip(test_scores, t_test))

        print 'Eval score', sum2
        targetSum2 += sum2
        targetSum += suml

    # Train final classifier
    clf.fit(training_data, targets, weights)
    try:
        joblib.dump(clf, 'train/{name}_clf.pkl'.format(name=clf.__class__.__name__), compress=9)
    except:
        print 'SAVING DOES NOT WORK NOT SURE WHY'

    print
    print 'Total sum2 per target', targetSum

    # targetSum /= float(len(training_data))
    # targetSum2 /= float(len(training_data))

    sumWeights = np.sum(weights) if weights else len(targets)

    targetSum /= sumWeights
    targetSum2 /= sumWeights

    targetSum2 = math.sqrt(targetSum2)

    print 'Normalised sqrt(sum2)', targetSum2
    print 'Normalised abs(..)', targetSum
    print

    # if doCrossVal:
    print 'Feature importances:'
    print clf.feature_importances_

    for i, imp in enumerate(clf.feature_importances_):
        print imp, trainVars[i] if i<len(trainVars) else 'N/A'
    
    return clf


def readFiles():
    print 'Reading files...'


    arr = root2rec('data/mvaTrainingSample.root', 'Flat')
    
    global trainVars

    trainVars = [
        'particleFlow_U', 'particleFlow_SumET', 'particleFlow_UPhi',
        'track_U', 'track_SumET', 'track_UPhi',
        'noPileUp_U', 'noPileUp_SumET', 'noPileUp_UPhi',
        'pileUp_MET', 'pileUp_SumET', 'pileUp_METPhi',
        'pileUpCorrected_U', 'pileUpCorrected_SumET', 'pileUpCorrected_UPhi',
        'jet1_pT', 'jet1_eta', 'jet1_Phi', 
        'jet2_pT', 'jet2_eta', 'jet2_Phi', 
        'nJets', 'numJetsPtGt30', 'nPV'
    ]

    print 'slicing variables'
    training = (np.asarray([arr[var] for var in trainVars])).transpose()

    # weightIndex = arrS.dtype.names.index('evtWeight')

    print 'slicing weights'
    weights = arr['weight']


    targets = arr['z_pT']
    # targets = arr['target_diffu']
    # targets = arr['target_phi']

    print 'Square sum PF:', math.sqrt(np.sum((arr['particleFlow_U'] - arr['z_pT'])**2)/len(arr))
    print 'Abs diff PF:', np.sum(abs(arr['particleFlow_U'] - arr['z_pT']))/len(arr)

    print 'Square sum Phi:', math.sqrt(np.sum((arr['target_phi'])**2)/len(arr))
    print 'Abs diff Phi:', np.sum(abs(arr['target_phi']))/len(arr)

    print 'Square sum PF weighted:', math.sqrt(np.sum(arr['weight']*(arr['particleFlow_U'] - arr['z_pT'])**2)/np.sum(arr['weight']))
    print 'Abs diff PF weighted:', np.sum(arr['weight']*abs(arr['particleFlow_U'] - arr['z_pT']))/np.sum(arr['weight'])

    print 'Square sum U weighted:', math.sqrt(np.sum(arr['weight']*(arr['target_diffu'])**2)/np.sum(arr['weight']))
    print 'Abs diff U weighted:', np.sum(arr['weight']*abs(arr['target_diffu']))/np.sum(arr['weight'])

    print 'Done reading files.'

    return training, weights, targets


if __name__ == '__main__':

    classifier = 'GBRT' # 'Ada' #'GBRT'
    doTrain = True
    useWeights = False


    # Settings for evaluation
    percentile = 83.5
    ntrees = 1919 # <- if best point is not at max tree size

    print 'Read training and test files...'
    training, weights, targets = readFiles()        

    print 'Sizes'
    print training.nbytes, weights.nbytes, targets.nbytes

    # clf = createSVR(C=0.1)
    # clf = createBaggingSVR()
    clf = createGBRT()

    if not useWeights:
        weights = None

    if doTrain:
        print 'Start training'
        train(clf, training, targets, weights)
    

