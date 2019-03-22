# Import packages
import matplotlib.pyplot as plt
import numpy as np
import math as mt
import scipy as sp
import pandas as pd
import os
from astropy.io import fits
from astropy.io import ascii
from astropy.cosmology import Planck15 as model
from sklearn.neighbors import BallTree
import random
import time
import csv
# Import calculation and plotting functions
import modules as md
import plotting as pot



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
file_directory		   = work_directory + "data_files/"
# Set the data file names from directory on computer
DR7_data_filename      = "post_catalog.dr72bright0.fits"
DR7_randoms_filename0  = "random-0.dr72bright.fits"
DR7_randoms_filename1  = "random-1.dr72bright.fits"
DR7_randoms_filename10 = "random-10.dr72bright.fits"

########################
# Read in the DR7 data #
########################
hdu1     = fits.open(data_directory + DR7_data_filename)
DR7_data = hdu1[1].data
RA_d     = DR7_data.field(1)
DEC_d    = DR7_data.field(2)
z_d      = DR7_data.field(9)
abs_m_d  = DR7_data.field(8)
mr       = [abs_m_d[i][2] for i in range(0,len(abs_m_d))]

###########################
# Read in the Random data #
###########################

hdu2        = fits.open(data_directory + DR7_randoms_filename0)
hdu3        = fits.open(data_directory + DR7_randoms_filename1)
hdu4        = fits.open(data_directory + DR7_randoms_filename10)
DR7_rands0  = hdu2[1].data
DR7_rands1  = hdu3[1].data
DR7_rands10 = hdu4[1].data
RA_r_0      = DR7_rands0.field(0)
DEC_r_0     = DR7_rands0.field(1)
RA_r_1      = DR7_rands1.field(0)
DEC_r_1     = DR7_rands1.field(1)
RA_r_10     = DR7_rands10.field(0)
DEC_r_10    = DR7_rands10.field(1)

RA_r        = np.append(RA_r_0, RA_r_1)
RA_r        = np.append(RA_r, RA_r_10)
DEC_r       = np.append(DEC_r_0, DEC_r_1)
DEC_r       = np.append(DEC_r, DEC_r_10)

# Check distance calculations:
RA_start  = 205
DEC_start = 10
window    = 1
RA_low    = RA_start
RA_high   = RA_start + window
DEC_low   = DEC_start
DEC_high   = DEC_start + window
# Set bins
bin_num = 10
bin_select = np.logspace(np.log10(0.1), np.log10(50), bin_num, endpoint=True)

########################
# Select data galaxies #
########################
(RA_1, DEC_1, z_1) = md.data_selector(RA_d, DEC_d, z_d,
							   RA_low, RA_high, DEC_low, DEC_high)
data_3vec = [[RA_1[i], DEC_1[i], z_1[i]] for i in range(0, len(RA_1))]
data_cartesian = np.array([md.spherical_to_cartesian(ii[0], ii[1], ii[2])
						   for ii in data_3vec])
data_len  = len(RA_1)
print("Number of data galaxies selected: ", data_len)

##########################
# Select Random Galaxies #
##########################
(RA_2, DEC_2, z_2) = md.random_selector(RA_r, DEC_r, z_1,
							   RA_low, RA_high, DEC_low, DEC_high)
rand_3vec = [[RA_2[i], DEC_2[i], z_2[i]] for i in range(0, len(RA_2))]
rand_cartesian = np.array([md.spherical_to_cartesian(ii[0], ii[1], ii[2])
						   for ii in rand_3vec])
rand_len  = len(RA_2)
print("Number of random galaxies selected: ", rand_len)
print("Ratio of rand/data: ", rand_len/data_len)

rat = rand_len/data_len
print("Ratio of rand/data: ", int(rat))

##################################
### ESTIMATOR STABILITY CHECKS ###
##################################
x_axis = (bin_select[1:] + bin_select[:-1])/2	# for plotting
factor_array = [1,2,3]#int(rat)]#[1,3,5,9,int(rat)]
print("Factors to be calculated: \n", factor_array, "\n")
estim_array1   = []
estim_array2   = []
estim_array3   = []
DD  = md.auto_sort_pair_counter(data_cartesian, bin_select, 0)
for ii in factor_array:
	start = time.time()
	# downsample random dataset
	(RA_2, DEC_2, z_2) = md.random_selector(RA_r, DEC_r, z_1,
								   RA_low, RA_high, DEC_low, DEC_high)
	rand_3vec_dwn = np.array(random.sample(rand_3vec, ii*len(RA_1)))
	rand_cart_dwn = np.array([md.spherical_to_cartesian(ii[0], ii[1], ii[2])
							   for ii in rand_3vec_dwn])
	"""
	#np.array(random.sample(rand_cartesian.tolist(), ii*len(RA_1)))
	print(data_cartesian[0:5])
	print(rand_cart_dwn[0:5])
	# plot positions and redshifts to check accuracy
	z_d = z_1
	z_r = []
	for jj in rand_3vec_dwn:
		z_r.append(jj[2])
	pot.z_histogram_3(z_d, z_r)
	RA_r  = []
	DEC_r = []
	for jj in rand_3vec_dwn:
		RA_r.append(jj[0])
		DEC_r.append(jj[1])
	RA_array  = [RA_1, RA_r]
	DEC_array = [DEC_1, DEC_r]
	pot.sky_positions_2(RA_array, DEC_array)
	plt.show()
	"""
	# size of data sets
	N_d = len(data_cartesian)
	N_r = len(rand_cart_dwn)
	# Calculate the pair counts
	RR  = md.auto_sort_pair_counter(rand_cart_dwn, bin_select, 0)
	DR  = md.improved_cross_correlate(data_cartesian, rand_cart_dwn, bin_select, 0)
	# calculate the estimators
	(xi_1, xi_2, xi_3) = md.estimator_calculator(N_d, N_r, DD, RR, DR, 3)
	end = time.time()
	# Display data
	print("Factor Multiple:", ii)
	print("time taken: " + str(end-start))
	print("data len = " + str(N_d))
	print("rand len = " + str(N_r))
	print("DD count = " + str(DD))
	print("RR count = " + str(RR))
	print("DR count = " + str(DR))
	print("Simple Davis & Peebles estimator per bin:" + "\n", xi_1)
	print("Davis & Peebles estimator per bin:" + "\n", xi_2)
	print("Landay & Szalay estimator per bin:" + "\n", xi_3)
	print("### \n")
	estim_array1.append(xi_1)
	estim_array2.append(xi_2)
	estim_array3.append(xi_3)

# Data File Names
f_name0 = file_directory + "dp0_est.csv"
f_name1 = file_directory + "dp1_est.csv"
f_name2 = file_directory + "lz_est.csv"

# Write Data to files
md.data_writer(x_axis, np.array(estim_array1), factor_array, f_name0)
md.data_writer(x_axis, np.array(estim_array2), factor_array, f_name1)
md.data_writer(x_axis, np.array(estim_array3), factor_array, f_name2)

# Read data from files
(mult_facs, bins, data_dps) = md.data_reader(f_name0)
(_, _, data_dp)				= md.data_reader(f_name1)
(_, _, data_lz) 			= md.data_reader(f_name2)

print(mult_facs)
print(bins)
print(data_dps)


# plot estimators
pot.estimator_plots(bins, data_dps, mult_facs, "Davis & Peebles Simple")
pot.estimator_plots(bins, data_dp, mult_facs, "Davis & Peebles ")
pot.estimator_plots(bins, data_lz, mult_facs, "Landay & Szalay")
plt.show()

exit()

"""
f = open("stability.txt", "w")
f.write(str(factor_array) + "\n")
for ii in estim_array1:
	f.write(str(ii) + "\n")
for ii in estim_array2:
	f.write(str(ii) + "\n")
for ii in estim_array3:
	f.write(str(ii) + "\n")
f.close()

g = open("stability.txt", "r")
data = g.readlines()
for dat in data[0:1]:
	d = dat[1:-2].strip()
	mults = [float(ii) for ii in (d.split(", "))]
runs = len(mults)
array_1 = []
array_2 = []
array_3 = []
for dat in data[1:1+runs]:
	d = dat[1:-2].strip()
	array_1.append([float(ii) for ii in (d.split(", "))])
for dat in data[1+runs:1+2*runs]:
	d = dat[1:-2].strip()
	array_2.append([float(ii) for ii in (d.split(", "))])
for dat in data[1+2*runs:1+3*runs]:
	d = dat[1:-2].strip()
	array_3.append([float(ii) for ii in (d.split(", "))])
print(array_3)
#print(data)
#print(data[0].split())
#print(data[1])
g.close()
"""




#print(dp_estim1[0].shape())

#plot the davis and peebles simple estimator
plt.figure(figsize=(15,7))
x_axis = (bin_select[1:] + bin_select[:-1])/2
#print(x_axis.shape())
for ii in range(0, len(mults)):
	plt.plot(x_axis, array_1[ii], linewidth=2, markersize=12,
			label=str(mults[ii]))#, label="D&P_simple")
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title("Davis & Peebles - Simple")
plt.xlabel("R [Mpc]")
plt.ylabel("Correlation Amplitude")
plt.legend()
plt.savefig(work_directory + "test_plots/davis_peebles_simple.png")

# plot the davis and peebles estimator
plt.figure(figsize=(15,7))
x_axis = (bin_select[1:] + bin_select[:-1])/2
#print(x_axis.shape())
for ii in range(0, len(mults)):
	plt.plot(x_axis, array_2[ii], linewidth=2, markersize=12,
			label=str(mults[ii]))#, label="D&P_simple")
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title("Davis & Peebles")
plt.xlabel("R [Mpc]")
plt.ylabel("Correlation Amplitude")
plt.legend()
plt.savefig(work_directory + "test_plots/davis_peebles.png")

# Landay and Szalay
plt.figure(figsize=(15,7))
x_axis = (bin_select[1:] + bin_select[:-1])/2
#print(x_axis.shape())
for ii in range(0, len(mults)):
	plt.plot(x_axis, array_3[ii], linewidth=2, markersize=12,
			label=str(mults[ii]))#, label="D&P_simple")
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title("Landay & Szalay")
plt.xlabel("R [Mpc]")
plt.ylabel("Correlation Amplitude")
plt.legend()
plt.savefig(work_directory + "test_plots/landay_szalay.png")
#plt.show()
