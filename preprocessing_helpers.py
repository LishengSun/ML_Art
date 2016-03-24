"""
Image processing functions 
output training matrix into csv files
"""
from PIL import Image
import numpy as np
import glob
import random
import cv2
import itertools
from skimage import color

def resize_im(im, new_size=(100, 100)):
	"""resize im (as ndarray)
	return ndarray"""
	im_resized = np.array(im.resize(new_size))
	return im_resized

def reduce_color_depth(im, to_depth=18):
	""" reduce color space of im (as ndarray)
	(ex: from 24-bit to 18-bit)
	return ndarray
	"""
	mode_to_bpp = {'1':1, 'L':8, 'P':8, 'RGB':24, 'RGBA':32, 'CMYK':32, 'YCbCr':24, 'I':32, 'F':32}
	color_depth = mode_to_bpp[im.mode]
	N = 2**(color_depth/3) / 2**(to_depth/3)
	# print N
	im_arr = np.array(im)
	# print im_arr
	# print '------------------------------------'
	im_arr_reduced = np.zeros(im_arr.shape)
	for l in range(im_arr.shape[0]):
		for c in range(im_arr.shape[1]):
			im_arr_reduced[l][c] = [ele/N*N+N/2 for ele in im_arr[l][c]]
	return np.uint8(im_arr_reduced)


def prepare_pixel_matrix(filepath, label, resize=True, reduce_color=False):
	"""
	turn resized (100*100) RGB image into a feature matrix
	return matrix of shape: [n_samples, n_features+1], where 1 repr the label
	label = class label, ex: Dali = 1, Van Gogh = -1
	"""
	X = []
	for im_file in glob.glob(filepath):
		# resize image
		im = Image.open(im_file)
		if resize:
			im = resize_im(im)
			im = Image.fromarray(im)
		if reduce_color:
			im = reduce_color_depth(im)
		# reshape to paste one feature after another
		im_reshape = np.array(im).reshape((1, -1))
 		X.append(im_reshape[0])
 	X = np.asarray(X) # convert to np.array
	pixel_matrix = np.ones((X.shape[0], X.shape[1]+1))*label # add label = 1 as the last feature
	pixel_matrix[:, :-1] = X
	return pixel_matrix

def prepare_pixel_matrix_grayscale(filepath, label, resize=True):
	"""
	turn resized (100*100) grayscale image into a feature matrix
	return matrix of shape: [n_samples, n_features+1], where 1 repr the label
	label = class label, ex: Dali = 1, Van Gogh = -1
	"""
	X = []
	for im_file in glob.glob(filepath):
		# resize image
		im = Image.open(im_file)
		if resize:
			im = resize_im(im)
		im_gray = color.rgb2gray(im)
		im_reshape = np.array(im_gray).reshape((1, -1))
 		X.append(im_reshape[0])
 	X = np.asarray(X) # convert to np.array
	pixel_matrix_grayscale = np.ones((X.shape[0], X.shape[1]+1))*label # add label = 1 as the last feature
	pixel_matrix_grayscale[:, :-1] = X
	return pixel_matrix_grayscale

def Dali_Van_training_pixel_24bit(data_dir):
	"""return a csv file containing Dali and Van_Gogh training data
	already shuffled
	100*100 image, color_depth = 24bit (non-reduced)
	"""
	training_data_Dali = prepare_pixel_matrix(data_dir+'Dali_painting/*.jpg', 1)
	training_data_Van = prepare_pixel_matrix(data_dir+'Van_Gogh_painting/*.jpg', -1)
	list_D = training_data_Dali.tolist()
	list_V = training_data_Van.tolist()
	training_data = np.asarray(list(itertools.chain.from_iterable(zip(list_D, list_V))))
	np.savetxt(data_dir+'training_Dali_Van_pixel_24bit.csv', training_data, delimiter=',')


def Dali_Van_training_pixel_18bit(data_dir):
	"""return a csv file containing Dali and Van_Gogh training data
	already shuffled
	100*100 image, color_depth = 24bit (non-reduced)
	"""
	training_data_Dali = prepare_pixel_matrix(data_dir+'Dali_painting/*.jpg', 1, reduce_color=True)
	training_data_Van = prepare_pixel_matrix(data_dir+'Van_Gogh_painting/*.jpg', -1, reduce_color=True)
	list_D = training_data_Dali.tolist()
	list_V = training_data_Van.tolist()
	training_data = np.asarray(list(itertools.chain.from_iterable(zip(list_D, list_V))))
	np.savetxt(data_dir+'training_Dali_Van_pixel_18bit.csv', training_data, delimiter=',')


def Dali_Van_training_grayscale(data_dir):
	"""return a csv file containing Dali and Van_Gogh training data
	already shuffled
	100*100 image, color_depth = 24bit (non-reduced)
	"""
	training_data_Dali = prepare_pixel_matrix_grayscale(data_dir+'Dali_painting/*.jpg', 1)
	training_data_Van = prepare_pixel_matrix_grayscale(data_dir+'Van_Gogh_painting/*.jpg', -1)
	list_D = training_data_Dali.tolist()
	list_V = training_data_Van.tolist()
	training_data = np.asarray(list(itertools.chain.from_iterable(zip(list_D, list_V))))
	np.savetxt(data_dir+'training_Dali_Van_pixel_grayscale0000000.csv', training_data, delimiter=',')


def Calculate_Color_Hist(filepath):
	"""
	compute the Color Histogram of all images in filepath
	and put them in an array of shape 
	n(# of samples)*m (768 = 256 bins * 3 chanels)
	"""
	bgr_Hist = []
	color = ('b', 'g', 'r')
	img_list = glob.glob(filepath)
	img_names = [n.split('/')[-1] for n in img_list]
	for im_file in img_list:
		img = cv2.imread(im_file)
		for i, col in enumerate(color):
			hist = cv2.calcHist([img], [i], None, [256], [0, 256])
			hist = hist / np.sum(hist) # histogram normalization
			bgr_Hist.append(hist)
	bgr_Hist = np.asarray(bgr_Hist).reshape(len(img_list), -1)
	bgr_Hist_named = np.column_stack((img_names, bgr_Hist))

	return bgr_Hist_named

def DaVinci_Botticelli_training_ColorHist(data_dir):
	"""
	create a training csv file containing 2 classes:
	1 for DaVinci, -1 for Botticelli
	return an ndarray: n*(m+1)
	n: number of samples
	m: number of features (color Histogram here)
	+1: the target column

	"""
	bgr_Hist_DaVinci = Calculate_Color_Hist(data_dir+'DaVinci/certain/*.jpg')
	bgr_Hist_Botticelli = Calculate_Color_Hist(data_dir+'Botticelli/*.jpg')
	target_DaVinci = np.ones(bgr_Hist_DaVinci.shape[0])
	training_DaVinci = np.column_stack((bgr_Hist_DaVinci, target_DaVinci))
	target_Botticelli = np.ones(bgr_Hist_Botticelli.shape[0])*(-1)
	training_Botticelli = np.column_stack((bgr_Hist_Botticelli, target_Botticelli))
	
	
	training = np.vstack((training_DaVinci, training_Botticelli))	
	np.random.shuffle(training)
	np.savetxt(data_dir+'training_DaVinci_Botticelli_ColorHist.csv', training, delimiter=',', fmt='%s')


if __name__ == "__main__":
	DaVinci_Botticelli_training_ColorHist('./DaVinci_Botticelli/data/')

