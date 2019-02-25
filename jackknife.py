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
work_directory         = "/Users/kianalexandershakerin/Desktop/fife/dissertation/"
data_directory         = work_directory + "catalogues/"
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
### 	 Jackknife Tests	   ###
##################################
# First we calculate the correlation function
# on the whole data set.
Nd = len(data_cartesian)
Nr = len(rand_cartesian)
DD = md.auto_sort_pair_counter(data_cartesian, bin_select, 0)
RR = md.auto_sort_pair_counter(rand_cartesian, bin_select, 0)
DR = md.improved_cross_correlate(data_cartesian, rand_cartesian, bin_select, 0)
(xi_1_m, xi_2_m, xi_3_m) = md.estimator_calculator(Nd, Nr, DD, RR, DR, 3)


# Next we split a set of data into N subvolumes
# and perform the jackknife error estimate
RA_sub    = 3
DEC_sub	  = 3
RA_subs  = np.linspace(RA_low, RA_high, RA_sub+1, endpoint=True)
DEC_subs = np.linspace(DEC_low, DEC_high, DEC_sub+1, endpoint=True)
print(RA_subs)
print(DEC_subs)
plot_subs_d = []
plot_subs_r = []
data_subs = []
rand_subs = []
# Loop over all subsections and select the data
for ii in range(RA_sub):
	RA_l = RA_subs[ii]
	RA_h = RA_subs[ii+1]
	for jj in range(DEC_sub):
		DEC_l = DEC_subs[jj]
		DEC_h = DEC_subs[jj+1]
		# Select the data corresponding to a subsection
		(RA_dj, DEC_dj, z_dj) = md.data_selector(RA_d, DEC_d, z_d,
									   RA_l, RA_h, DEC_l, DEC_h)
		data_3vec = [[RA_dj[xx], DEC_dj[xx], z_dj[xx]] for xx in range(0, len(RA_dj))]
		data_cartesian = np.array([md.spherical_to_cartesian(yy[0], yy[1], yy[2])
								   for yy in data_3vec])
		plot_subs_d.append([RA_dj, DEC_dj, z_dj])
		data_subs.append(data_cartesian)
	    # Select the randoms corresponding to a subsection
		(RA_rj, DEC_rj, z_rj) = md.random_selector(RA_r, DEC_r, z_dj,
									   RA_l, RA_h, DEC_l, DEC_h)
		rand_3vec = [[RA_rj[xx], DEC_rj[xx], z_rj[xx]] for xx in range(0, len(RA_rj))]
		rand_cartesian = np.array([md.spherical_to_cartesian(yy[0], yy[1], yy[2])
								   for yy in rand_3vec])
		plot_subs_r.append([RA_rj, DEC_rj, z_rj])
		rand_subs.append(rand_cartesian)

# Plot sky positions of subdivisions
plt.figure(figsize=(15,7))
plt.scatter(RA_1, DEC_1, marker='o', s=3)
for ii in plot_subs_d:
	plt.scatter(ii[0], ii[1], marker='o', s=1)
plt.title("Galaxy Catelogue Sky Positions")
plt.xlabel("RA [deg]")
plt.ylabel("DEC [deg]")
#plt.show()


xi_1_store = []
xi_2_store = []
xi_3_store = []
for ii in range(0, len(data_subs)):
	print("N: " + str(ii))
	data_hold = []
	rand_hold = []
	for jj in range(0, len(data_subs)):
		if (jj == ii):
			continue
		data_hold = data_hold + data_subs[jj].tolist()
		rand_hold = rand_hold + rand_subs[jj].tolist()
	#print(data_hold[0:3])
	Nd = len(data_hold)
	Nr = len(rand_hold)
	DD = md.auto_sort_pair_counter(np.array(data_hold), bin_select, 0)
	RR = md.auto_sort_pair_counter(np.array(rand_hold), bin_select, 0)
	DR = md.improved_cross_correlate(np.array(data_hold), np.array(rand_hold), bin_select, 0)
	(xi_1, xi_2, xi_3) = md.estimator_calculator(Nd, Nr, DD, RR, DR, 3)
	xi_1_store.append(xi_1)
	xi_2_store.append(xi_2)
	xi_3_store.append(xi_3)


xi_1_avg = md.big_average(xi_1_store)
xi_2_avg = md.big_average(xi_2_store)
xi_3_avg = md.big_average(xi_3_store)
#print(xi_1_avg)
#print(xi_2_avg)
#print(xi_3_avg)

# Calculate the covariance
(cv_matrix, diags) = md.covariance_matrix(xi_1_store, RA_sub*DEC_sub, 9)
print(pd.DataFrame(data=cv_matrix))
print(diags)

x_axis = (bin_select[1:] + bin_select[:-1])/2	# for plotting
plt.figure()
plt.errorbar(x_axis, xi_1_m, yerr=diags)
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title("D&P Simple Estimator")
plt.show()
