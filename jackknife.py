# Import packages
import matplotlib.pyplot as plt
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
DR7_data_filename      = "post_catalog.dr72bright0.fits"
DR7_randoms_filename0  = "random-0.dr72bright.fits"
DR7_randoms_filename1  = "random-1.dr72bright.fits"
DR7_randoms_filename10 = "random-10.dr72bright.fits"
MaNGA_targets_filename = "MaNGA_targets_extNSA_tiled_ancillary.fits"

# Imports the DR7 RA/DEC/z/mag
(RA_d, DEC_d, z_d, mr) = md.DR7_data_importer(data_directory + DR7_data_filename)
# Read in the Random data
filenames = [data_directory + DR7_randoms_filename0,
			 data_directory + DR7_randoms_filename1,
			 data_directory + DR7_randoms_filename10]
(RA_r, DEC_r) = md.DR7_randoms_importer(filenames)
# Read in the MaNGA Target Catalogue data
(RA_mng, DEC_mng, z_mng, mr_mng) = md.MANGA_data_importer(data_directory + MaNGA_targets_filename)
# Set the RA/DEC window for galaxy observations
(RA_low, RA_high, DEC_low, DEC_high) = md.initialize_window(180, 10, 2, 2)
# Set bins
bin_num = 10
bin_select = np.logspace(np.log10(0.1), np.log10(50), bin_num, endpoint=True)


# Select the DATA and the RANDOM galaxies
dat = np.array([[RA_d[ii], DEC_d[ii], z_d[ii], mr_mng[ii]] for ii in range(0, len(RA_d))])
ran = np.array([RA_r[ii], DEC_r[ii]] for ii in range(0,len(RA_r))]
win = [RA_low, RA_high, DEC_low, DEC_high]
(data_3vec, data_cartesian, rand_3vec, rand_cartesian) = md.setup_data_and_randoms(dat, ran, win)

# Select the MaNGA galaxies
dat_mang = [RA_mng, DEC_mng, z_mng]
(manga_3vec, manga_cartesian, rand_mang_3vec, rand_mang_z_match) = md.setup_data_and_randoms(dat_mang, ran, win)


# AFC Jackknife function
subdivisions = [6,6]

(ACF, ACF_s, covar, e_bars) = md.ACF_jackknife(data_3vec, rand_3vec,
											   bin_select, win,
											   subdivisions, 1)
print("Everything from the function")
print(ACF)
print(ACF_s)
print(pd.DataFrame(data=covar))
print(e_bars)


x_axis = (bin_select[1:] + bin_select[:-1])/2	# for plotting
plt.figure()
plt.errorbar(x_axis, ACF, yerr=e_bars, label="function", color="r")
#plt.gca().set_xscale("log")
#plt.gca().set_yscale("log")
plt.title("ACF With Jackknife Error Bars")
plt.show()

exit()

##################################
### 	 Jackknife Tests	   ###
##################################

# on the whole data set.
Nd = len(data_cartesian)
Nr = len(rand_cartesian)
DD = md.auto_sort_pair_counter(data_cartesian, bin_select, 0)
RR = md.auto_sort_pair_counter(rand_cartesian, bin_select, 0)
DR = md.improved_cross_correlate(data_cartesian, rand_cartesian, bin_select, 0)
(xi_1_m, xi_2_m, xi_3_m) = md.estimator_calculator(Nd, Nr, DD, RR, DR, 3)
#print(xi_2_m)

RA_sub    = 2
DEC_sub	  = 2
RA_subs  = np.linspace(RA_low, RA_high, RA_sub+1, endpoint=True)
DEC_subs = np.linspace(DEC_low, DEC_high, DEC_sub+1, endpoint=True)
#print(RA_subs)
#print(DEC_subs)
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


"""
# Plot sky positions of subdivisions
plt.figure(figsize=(15,7))
plt.scatter(data_3vec[:,0], data_3vec[:,1], marker='o', s=3)
for ii in plot_subs_d:
	plt.scatter(ii[0], ii[1], marker='o', s=1)
plt.title("Galaxy Catelogue Sky Positions")
plt.xlabel("RA [deg]")
plt.ylabel("DEC [deg]")
#plt.show()
"""

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
	#RR = md.auto_sort_pair_counter(np.array(rand_hold), bin_select, 0)
	DR = md.improved_cross_correlate(np.array(data_hold), np.array(rand_hold), bin_select, 0)
	xi_2 = md.davis_peebles(DD, DR, Nd, Nr)
	#(xi_1, xi_2, xi_3) = md.estimator_calculator(Nd, Nr, DD, RR, DR, 3)
	#xi_1_store.append(xi_1)
	xi_2_store.append(xi_2)
	#xi_3_store.append(xi_3)

print(xi_2_store)

#xi_1_avg = md.big_average(xi_1_store)
#xi_2_avg = md.big_average(xi_2_store)
#xi_3_avg = md.big_average(xi_3_store)
#print(xi_1_avg)
#print(xi_2_avg)
#print(xi_3_avg)

# Calculate the covariance
(cv_matrix, diags) = md.covariance_matrix(xi_2_store, RA_sub*DEC_sub, 9)
print(pd.DataFrame(data=cv_matrix))
print(diags)

x_axis = (bin_select[1:] + bin_select[:-1])/2	# for plotting
plt.figure()
plt.errorbar(x_axis, ACF, yerr=e_bars, label="function", color="r")
plt.errorbar(x_axis, xi_2_m, yerr=diags, label="original", color="b")
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title("ACF With Jackknife Error Bars")
plt.show()
