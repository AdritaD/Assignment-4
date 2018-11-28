from sklearn.decomposition import PCA
from pylab import *
from skimage import data, io, color
import sys
import cv2
import numpy as np

img_list = sys.argv[1:6]
num = 1
for i in img_list:

	link = "original_images/"+i
	image_gray = io.imread(link,as_gray=True)


	n_comp=[5,25,125]
	for j in n_comp:
		pca = PCA(n_components = j)
		pca.fit(image_gray)
		image_gray_pca = pca.fit_transform(image_gray)
		image_gray_restored = pca.inverse_transform(image_gray_pca)

		io.imshow(image_gray_restored)
		xlabel('Restored image n_components = %s' %n_comp)

		show()