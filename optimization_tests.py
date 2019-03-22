# Import packages
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import numpy as np
import math as mt
import scipy as sp
import pandas as pd
import os
from astropy.io import fits
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy import units as uni
from astropy.cosmology import Planck15 as model
from sklearn.neighbors import BallTree
import random
import time
import csv
# Import calculation and plotting functions
import modules as md
import plotting as pot
import pprint

#############################################################
#-----------------------------------------------------------#
#-----------------------------------------------------------#
#                        MAIN BODY                          #
#-----------------------------------------------------------#
#-----------------------------------------------------------#
#############################################################
# Set the directories in which the data resides
work_directory         = "/Users/kianalexandershakerin/Desktop/fife/dissertation/clustering_analysis/"
data_directory         = work_directory + "catalogues/"
# Set the data file names from directory on computer
DR7_data_filename         = "post_catalog.dr72bright0.fits"
DR7_randoms_filename0     = "random-0.dr72bright.fits"
DR7_randoms_filename1     = "random-1.dr72bright.fits"
DR7_randoms_filename10    = "random-10.dr72bright.fits"
MaNGA_targets_filename    = "MaNGA_targets_extNSA_tiled_ancillary.fits"
MaNGA_kinematics_filename = "mpl6_basic_kinematic_info.csv" #NOTE: CSV
MaNGA_properties_filename = "manga_firefly-v2_4_3-GLOBALPROP.fits"

# Imports the DR7 RA/DEC/z/mag
(RA_d, DEC_d, z_d, mr) = md.DR7_data_importer(data_directory + DR7_data_filename)
# Read in the Random data
filenames = [data_directory + DR7_randoms_filename0,
			 data_directory + DR7_randoms_filename1,
			 data_directory + DR7_randoms_filename10]
(RA_r, DEC_r) = md.DR7_randoms_importer(filenames)
# Read in the MaNGA Target Catalogue data
(RA_mng, DEC_mng, z_mng, mr_mng, corr) = md.MANGA_data_importer(data_directory + MaNGA_targets_filename)
#print(corr[0:5])
# Set the RA/DEC window for galaxy observations
(RA_low, RA_high, DEC_low, DEC_high) = md.initialize_window(180, 10, 5, 5)
win 								 = [RA_low, RA_high, DEC_low, DEC_high]
# Set bins
bin_num    = 15
bin_select = np.logspace(np.log10(0.1), np.log10(50), bin_num, endpoint=True)
x_axis     = (bin_select[1:] + bin_select[:-1])/2

manga_data = np.array([[RA_mng[ii], DEC_mng[ii], z_mng[ii]] for ii in range(0, len(RA_mng))])
DR7_data = np.array([[RA_d[ii], DEC_d[ii], z_d[ii]] for ii in range(0, len(RA_d))])
random   = np.array([[RA_r[ii], DEC_r[ii]] for ii in range(0, len(RA_r))])
(data_sp, data_c, rand_sp, rand_c) = md.setup_data_and_randoms(DR7_data, random, win)
(manga_sp, manga_c, mrand_sp, mrand_c) = md.setup_data_and_randoms(manga_data, random, win)

print("Length of MaNGA data: " + str(len(manga_c)))
print("Length of Random data: " + str(len(mrand_c)))

#test_data = data_c.tolist()
#test_data = test_data[0:15]
#print(test_data)
# NOTE: Code adapted from https://www.youtube.com/watch?v=XqXSGSKc8NU&t=218s
# k-d tree constuction

def build_kdtree(data_set, max_depth, depth=0, kdim=3):
	n 			  = len(data_set)	# length of data set
	if (n <= 0):					# if no data points return none
		return None
	if (depth >= max_depth):
		return data_set
	axis 		  = depth % kdim		# splitting axis
	sorted_points = sorted(data_set, key=lambda point: point[axis])
	return {
		'point': sorted_points[int(n/2)],
		'left': build_kdtree(sorted_points[:int(n/2)], max_depth, depth + 1),
		'right': build_kdtree(sorted_points[int(n/2 + 1):], max_depth, depth + 1)
	}


def pair_counter(dset, point, limits, axis):
	dset = np.array(dset)
	#print(dset)
	#print(point[axis])
	#print(dset[:,axis])
	r1, r2 = limits
	pairs  = 0
	dset = np.array(dset)
	if dset is None:
		return 0
	#sorted_points = sorted(data_set, key=lambda point: point[axis])
	#dset = np.array(dset)
	select_data = dset[np.where((r1 < abs(dset[:,axis]-point[axis])) &
							    (abs(dset[:,axis]-point[axis]) < r2))]
	if len(select_data) <= 0:
		return 0

	for data in select_data:
		distance = md.distance_cartesian(point, data)
		if (r1 < distance < r2):
			pairs += 1

	return pairs


def kdtree_search(root, point, limits, max_depth, depth=0, kdim=3):
	r1, r2 = limits
	total_pairs = 0
	if root is None:	# Returns None if there are no points
		return 0
	axis 			= depth % kdim	# Find splitting axis
	#print("depth",depth, "axis",axis)
	next_branch 	= None
	opposite_branch = None

	if depth != (max_depth):
		if point[axis] < root['point'][axis]:
			next_branch 	= root['left']
			opposite_branch = root['right']
		else:
			next_branch 	= root['right']
			opposite_branch = root['left']
		if (r1 < md.distance_cartesian(point, root['point']) < r2):
			total_pairs += 1
		if (abs(point[axis] - root['point'][axis]) < r2):
			total_pairs += kdtree_search(next_branch, point, limits, max_depth, depth + 1)
			total_pairs += kdtree_search(opposite_branch, point, limits, max_depth, depth + 1)
		else:
			total_pairs += kdtree_search(next_branch, point, limits, max_depth, depth + 1)
	else:
		axis 		 = (depth-1) % 3
		total_pairs += pair_counter(root, point, limits, axis)

	return total_pairs




def kd_tree_pair_counter(dataset1, dataset2, bins, tree_depth):
	# Begin by constructing tree for depth as defined in arguments
	# dset 1 corresponds to the smaller data set
	# dset 2 corresponds to the larger data set
	dset1      = dataset1.tolist()	# need to conver arrays to lists
	dset2      = dataset2.tolist()	# need to conver arrays to lists
	kdtree     = build_kdtree(dset2, tree_depth)
	# Next we loop over all the data
	bin_counts   = []
	for ii in range(0, len(bins)-1):
		bin = [bins[ii], bins[ii+1]]
		total_pairs = 0
		for data in dataset1:
			counts 		 = kdtree_search(kdtree, data, bin, tree_depth)
			total_pairs += counts
		bin_counts.append(total_pairs)
	return bin_counts

# Run the k-d tree test here
tree_depth = 17
print("Tree Depth: ", tree_depth)
start = time.time()

test_data = manga_c
rand_data = mrand_c
#kdtree    = build_kdtree(rand_data, tree_depth)
kd_counts = kd_tree_pair_counter(test_data, rand_data, bin_select, tree_depth)

end = time.time() - start
print("Bin Counts:")
print(kd_counts)
print("Total time taken for k-d tree search: " + str(end))


# Run a comparison test with existing ACF and CCFs
# ACF
start = time.time()
#ACF = md.auto_sort_pair_counter(data_c, bin_select, 1)
CCF  = md.improved_cross_correlate(manga_c, mrand_c, bin_select, 0)
end = time.time() - start
print("ACF module pair count:")
print(CCF)
print("Total time taken for ACF: " + str(end))

"""
# Most basic crude cross correlation
start = time.time()
bin_counts   = []
for ii in range(0, len(bin_select)-1):
	bin = [bin_select[ii], bin_select[ii+1]]
	CCF = md.crude_cross_correlate(data_c, rand_c, bin[0], bin[1])
	bin_counts.append(CCF)
end = time.time() - start
print("ACF module pair count:")
print(bin_counts)
print("Total time taken for ACF: " + str(end))
"""

exit()

threshold = 50
def split(data, threshold):
    size = len(data)
    if (size < threshold):
        return data
    else:
        median_v = np.median(data[:,0])
        median_i = np.where(data[:,0] == median_v)[0]
        index    = median_i[0]
        data_1   = data[0:index]
        data_2   = data[index::]
        l1       = len(data_1)
        l2       = len(data_2)
        print("length of node a = ", l1)
        print("length of node b = ", l2)
        node_1   = split(data_1, threshold)
        node_2   = split(data_2, threshold)
        root     = [node_1, node_2, median_v]
        return root

root = split(data_c, 200)
print(root)
"""
   function construct_balltree is
       input:
           D, an array of data points
       output:
           B, the root of a constructed ball tree
       if a single point remains then
           create a leaf B containing the single point in D
           return B
       else
           let c be the dimension of greatest spread
           let p be the central point selected considering c
           let L,R be the sets of points lying to the left and right of the median along dimension c
           create B with two children:
               B.pivot = p
               B.child1 = construct_balltree(L),
               B.child2 = construct_balltree(R),
               let B.radius be maximum distance from p among children
           return B
       end if
   end function
"""
"""
print("total data length = ", len(data_c))
median_v = np.median(data_c[:,0])
median_i = np.where(data_c[:,0] == median_v)[0]
index    = median_i[0]
node_a   = data_c[0:index]
node_b   = data_c[index::]
print("length of node a = ", len(node_a))
print("length of node b = ", len(node_b))
"""
# End #
