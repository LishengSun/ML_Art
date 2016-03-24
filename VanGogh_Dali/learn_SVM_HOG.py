import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color, exposure, data
from PIL import Image
import numpy as np 
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation
import time

"""
classification using HOG as feature and linear SVM as classifier
"""
start_time = time.time()

# load training images (grayscale)
print 'loading data...'
training_data = np.genfromtxt('./data/training_Dali_Van_pixel_grayscale.csv', delimiter=',')

X = training_data[:, :-1]
y = training_data[:, -1]

# compute the Histogram of Orientation Gradiant
print 'computing HOG ...'
HOG = []
for im in X:
	im = im.reshape((100, -1))
	fd, hog_image = hog(im, orientations=8, pixels_per_cell=(5, 5), \
		cells_per_block=(2, 2), visualise=True)
	HOG.append(fd)
HOG = np.asarray(HOG)
print 'number of features before: ', HOG.shape[1]

# use linear SVM for feature selection to reduce the size of HOG
print 'Extracting good HOG...'
lsvc = LinearSVC(C=1, penalty='l1', dual=False).fit(HOG, y)
model = SelectFromModel(lsvc, prefit=True)
HOG_new = model.transform(HOG)
print 'number of features after: ', HOG_new.shape[1]
np.savetxt('./data/training_Dali_Van_HOG.csv', HOG_new, delimiter=',')

# evaluation with SVM (linear kernel)
print 'Training and predicting with linear SVM...'
clf = SVC(kernel='linear', C=1, probability=True)
cv = cross_validation.ShuffleSplit(HOG_new.shape[0], n_iter=100, test_size=0.1, random_state=0)
train_sizes, train_scores, test_scores = learning_curve(clf, HOG_new, y, cv=cv, n_jobs=4, train_sizes=np.linspace(0.1, 1.0, 10))
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
print train_sizes
print test_scores_mean

print 'totle running time: %s seconds' %(time.time()-start_time)
plt.title("Learning Curve: Linear SVM (HOG)")
plt.xlabel("training size m")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
plt.plot(train_sizes, train_scores_mean, 'r*-', label="Training score")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
plt.plot(train_sizes, test_scores_mean, 'g*-', label="Cross-validation score")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="g")
plt.legend(loc="best")
plt.savefig('./plots/best_result_HOG.png')
