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
(RA_low, RA_high, DEC_low, DEC_high) = md.initialize_window(170, 20, 2, 2)
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

###########################
#  Select MaNGA Galaxies  #
###########################
(RA_3, DEC_3, z_3) = md.data_selector(RA_mng, DEC_mng, z_mng,
							   RA_low, RA_high, DEC_low, DEC_high)
manga_3vec         = [[RA_3[ii], DEC_3[ii], z_3[ii]] for ii in range(0, len(RA_3))]
manga_cartesian    = np.array([md.spherical_to_cartesian(ii[0], ii[1], ii[2])
						      for ii in manga_3vec])


# Split SDSS DR7 data on magnitude and test correlation
cut = -20.5
mag_data = np.array(md.data_selector_prop(RA_d, DEC_d, z_d, mr, RA_low, RA_high, DEC_low, DEC_high))
bright = mag_data[np.where(mag_data[:,3] < cut)]
dim    = mag_data[np.where(mag_data[:,3] >= cut)]
print(len(bright), len(dim))

# convert the bright and dim data sets into cartesian coordinates
bright_cart = np.array([md.spherical_to_cartesian(bright[ii,0],bright[ii,1],bright[ii,2]) + [bright[ii,3]]
			for ii in range(0, len(bright))])
dim_cart    = np.array([md.spherical_to_cartesian(dim[ii,0],dim[ii,1],dim[ii,2]) + [dim[ii,3]]
			   for ii in range(0, len(dim))])

# build a random catalogue to match the redshifts of the brights
(RA_bt, DEC_bt, z_bt) = md.random_selector(RA_r, DEC_r, bright[:,2],
							               RA_low, RA_high, DEC_low, DEC_high)
bt_rand_3vec = [[RA_bt[i], DEC_bt[i], z_bt[i]] for i in range(0, len(RA_bt))]
bt_rand_cart = np.array([md.spherical_to_cartesian(ii[0], ii[1], ii[2])
						   for ii in bt_rand_3vec])
# build a random catalogue to match the redshifts of the dims
(RA_dm, DEC_dm, z_dm) = md.random_selector(RA_r, DEC_r, dim[:,2],
							               RA_low, RA_high, DEC_low, DEC_high)
dm_rand_3vec = [[RA_dm[i], DEC_dm[i], z_dm[i]] for i in range(0, len(RA_dm))]
dm_rand_cart = np.array([md.spherical_to_cartesian(ii[0], ii[1], ii[2])
						   for ii in dm_rand_3vec])


#print(mr_mng)
"""
plt.figure(figsize=(15,7))
plt.scatter(z_d, mr, marker="o", s=1, color="b")
plt.scatter(z_mng, mr_mng, marker="o", s=1, color="r")
plt.title("entire set")
plt.show()

exit()
"""

# Do the correlation function for the bright subsample
nr_bt 	    = len(bt_rand_cart)
nd_bt 	    = len(bright_cart)
DD          = md.auto_sort_pair_counter(bright_cart[:,0:3], bin_select, 0)
RR          = md.auto_sort_pair_counter(bt_rand_cart, bin_select, 0)
DR          = md.improved_cross_correlate(bright_cart[:,0:3], bt_rand_cart, bin_select, 0)
bright_est1 = md.davis_peebles_simple(DD, RR, nd_bt, nr_bt)
bright_est2 = md.davis_peebles(DD, DR, nd_bt, nr_bt)
bright_est3 = md.landay_szalay(DD, RR, DR, nd_bt, nr_bt)
print("Bright:")
print("DD: " + str(DD))
print("RR: " + str(RR))
print("DR: " + str(DR))
print(bright_est1)
print(bright_est2)

# Do the correlation function for the dim sample
nr_dm	   = len(dm_rand_cart)
nd_dm 	   = len(dim_cart)
DD  	   = md.auto_sort_pair_counter(dim_cart[:,0:3], bin_select, 0)
RR  	   = md.auto_sort_pair_counter(dm_rand_cart, bin_select, 0)
DR         = md.improved_cross_correlate(dim_cart[:,0:3], dm_rand_cart, bin_select, 0)
dim_est1   = md.davis_peebles_simple(DD, RR, nd_dm, nr_dm)
dim_est2   = md.davis_peebles(DD, DR, nd_dm, nr_dm)
dim_est3   = md.landay_szalay(DD, RR, DR, nd_dm, nr_dm)
print("Dim:")
print("DD: " + str(DD))
print("RR: " + str(RR))
print("DR: " + str(DR))
print(dim_est1)
print(dim_est2)
print(dim_est3)

x_axis = (bin_select[1:] + bin_select[:-1])/2
plt.figure(figsize=(15,7))
plt.plot(x_axis, bright_est1, linewidth=2, markersize=12, label="bright")
plt.plot(x_axis, dim_est1, linewidth=2, markersize=12, label="dim")
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title("Magnitude Cut Correlation Function (Davis and Hamilton) Mag <= " + str(cut))
plt.xlabel("R [Mpc]")
plt.ylabel("Correlation Amplitude")
plt.legend()

plt.figure(figsize=(15,7))
plt.plot(x_axis, bright_est2, linewidth=2, markersize=12, label="bright")
plt.plot(x_axis, dim_est2, linewidth=2, markersize=12, label="dim")
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title("Magnitude Cut Correlation Function (Davis and Peebles) Mag > " + str(cut))
plt.xlabel("R [Mpc]")
plt.ylabel("Correlation Amplitude")
plt.legend()

plt.figure(figsize=(15,7))
plt.plot(x_axis, bright_est3, linewidth=2, markersize=12, label="bright")
plt.plot(x_axis, dim_est3, linewidth=2, markersize=12, label="dim")
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title("Magnitude Cut Correlation Function (Landay & Szalay) Mag > " + str(cut))
plt.xlabel("R [Mpc]")
plt.ylabel("Correlation Amplitude")
plt.legend()



plt.figure(figsize=(15,7))
#plt.scatter(mag_data[:,2], mag_data[:,3], marker="o", s=1, color="b")
plt.scatter(bright[:,2], bright[:,3], marker="x", s=1, color="r")
plt.scatter(dim[:,2], dim[:,3], marker=".", s=1, color="g")
plt.title("selection")


plt.show()
exit()
