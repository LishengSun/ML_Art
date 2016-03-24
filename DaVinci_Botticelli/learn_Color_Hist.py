import pandas as pd 
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation
from sklearn.metrics import make_scorer, roc_auc_score
import time, random
from unbalanced_dataset.over_sampling import SMOTE, OverSampler
from sklearn import tree
from sklearn.naive_bayes import GaussianNB

"""
classification using color histograms as feature
"""
startime = time.time()
print 'loading data...'
training = pd.read_csv('./data/training_DaVinci_Botticelli_ColorHist.csv', header=None)
names = training.iloc[:, 0]
targets = training.iloc[:, -1]
Feat_CH = training.iloc[:, 1:-1]

X = np.asarray(Feat_CH)
y = np.asarray(targets)
print 'number of training examples: Da Vinci: %s, Botticelli: %s'\
%(len([yy for yy in y if yy==1]), len([yy for yy in y if yy==-1]))

print 'number of features before: ', X.shape[1]
print 'feature selection via Linear SVM...'
lsvc = LinearSVC(C=100, penalty='l1', dual=False).fit(X, y) 
# according the validation curve (not output here), C=10 gives the best result
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
print 'number of features after: ', X_new.shape[1]


# Use SMOTE to 'fix' the imbalanced problem:
# the python implementation of SMOTE comes from
# https://github.com/fmfn/UnbalancedDataset/tree/master/unbalanced_dataset 
ratio = float(len([t for t in y if t==-1]))/float(len([t for t in y if t==1]))
# oversampler = OverSampler(ratio = ratio-1)
smote = SMOTE(k=3, ratio = ratio-1)  
smote.x = X_new
smote.y = y
smote.minc = 1
smote.maxc = -1
smote.ucd ={1: len([tg for tg in y if tg==1]), -1: len([tg for tg in y if tg==-1])}
ret_X, ret_y = smote.resample()
# overX, overy = oversampler.resample()

combined = zip(ret_X, ret_y)
random.shuffle(combined)
ret_X[:], ret_y[:] = zip(*combined)

print 'shuffled??\n', ret_y
print 'training and predicting...'
# clf = SVC(kernel='linear', C=1, probability=True)
# clf = tree.DecisionTreeClassifier()
clf = GaussianNB()
X_tr, X_test, y_tr, y_test = \
cross_validation.train_test_split(X_new, y, test_size=0.2, random_state=0)
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_test)#[:, 1]
print 'score sans resampling: ', roc_auc_score(y_test, y_pred)

ret_X_tr, ret_X_test, ret_y_tr, ret_y_test = \
cross_validation.train_test_split(ret_X, ret_y, test_size=0.2, random_state=0)
clf.fit(ret_X_tr, ret_y_tr)
ret_y_pred = clf.predict(ret_X_test)#[:, 1]
print 'score avec resampling: ', roc_auc_score(ret_y_test, ret_y_pred)


cv = cross_validation.ShuffleSplit(ret_X.shape[0], n_iter=100, test_size=0.2, random_state=0)
roc_scorer = make_scorer(roc_auc_score)#, needs_proba = True)
train_sizes, train_scores, test_scores = learning_curve(clf, ret_X, ret_y, cv=cv, scoring=roc_scorer, n_jobs=4, train_sizes=np.linspace(0.1, 1.0, 10))
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
print train_sizes
print test_scores_mean
print 'total running time: %s sec' %(time.time()-startime)

plt.title("Learning Curve: Gaussian Naive Bayes (Color Hist)")
plt.xlabel("training size m")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
plt.plot(train_sizes, train_scores_mean, 'r*-', label="Training score")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
plt.plot(train_sizes, test_scores_mean, 'g*-', label="CV score")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="g")
plt.legend(loc="best")
plt.savefig('./plots/GNB_CH_selC100.png')
plt.show()


