# ML_Art
Can machines learn art?

## Introduction
I am always passionate about the applications of Machine learning on art. <br /> This repository will record my findings on this subject. 
It is a project in progress with no pre-known destination. I will show my work in small folders, each treats a sub-problem.
<br />

## [Van_Gogh and Dali](../VanGogh_Dali)
After reading [this paper](http://cs229.stanford.edu/proj2010/BlessingWen-UsingMachineLearningForIdentificationOfArtPaintings.pdf), 
I decided to start by writing an algorithm in Python to identify the paintings of Van Gogh and those of Dali. 
Using the HOG features suggested by the paper, a 98.77% accuracy can be achieved with ~700 training example. This fits well the 
results presented in the paper. <br />

### Implementations:
- Training data: I used 700 examples (350 for Van Gogh, 350 for Dali) download from google image
- Classification algorithm: SVM (kernel=linear, C=1)
- Features: Histogram of Oriented Gradients (HOG)
- Features selection: LinearSVC-based
- [best accuracy](../VanGogh_Dali/plots/best_result_HOG.png): 98.77%
**Note**: the file 'training_Dali_Van_pixel_grayscale.csv' used in learn_SVM_HOG.py shoud be created by runining preprocessing_helpers.py. I didn't include it because it's too big.
<br />

### Future work:
To understand why HOG works so well in this problem, I consider to write a function to plot the selected-HOG features
over the image, which will allow me to know what exactly the machine 'sees'.

## [DaVinci and Botticelli](../DaVinci_Botticelli) (In progress and will update 'in real time')
Encouraged by this amazing result with Van Gogh and Dali, I tried to identify Da Vinci's work from Botticelli's. 
This time, the difficulties are: <br />
- Being two great painters of the same period - the Renaissance, no big style distinction can be suspected easily. <br />
- Devoting all his life in painting, Botticelli has left much more works than Da Vinci, for whom painting was only a part of his
great intelligence. Translated into machine learning words, this will be an imbalanced-data problem.<br />

<br />
### My first try: 

- Training data: 28 images of Da Vinci's paintings and 98 of Botticelli's from wikipedia's list. 
**Note**: I devided Da Vinci's paintings into 3 parts: those [generally accepted] (../DaVinci_Botticelli/data/DaVinci/certain) (28) as his work, those [might be] (../DaVinci_Botticelli/data/DaVinci/maybe) his work (6) 
and those [copied] (../DaVinci_Botticelli/data/DaVinci/copy) (3) by others. For this first try, I used only the first part.
- Features: color histogram (for instance, color is the only difference between the painters that I can find and 'translate' into calculatable
features)
- Feature engineering: SMOTE (to balance the data) + LinearSVC based feature selection
- Performance measure: AUC of ROC (to facilate the comparison of before and after SMOTE)
- classifiers tried: SVM, decision trees (DT), Gaussian Naive Bayes (GNB) <br />
<br />
##### Results and discussion: 

- SVM gives random guess
- DT and GNB can predict with different labels, but always suffering from [high bias](../DaVinci_Botticelli/plots/) even
I increase feature numbers by reducing the regularization of the feature selection. This suggests me to use more different features, 
for example, HOG, SIFT, ...

## References:
1. Van Gogh and Dali identification: [Using Machine Learning for Identification of Art Paintings](http://cs229.stanford.edu/proj2010/BlessingWen-UsingMachineLearningForIdentificationOfArtPaintings.pdf)
2. HOG: [Histograms of Oriented Gradients for Human Detection] (https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)
3. SMOTE: [SMOTE: Synthetic Minority Over-sampling Technique](https://www.jair.org/media/953/live-953-2037-jair.pdf) and [UnbalancedDataset](https://github.com/fmfn/UnbalancedDataset)
4. Feature selection: [An Introduction to Variable and Feature Selection](http://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf)




