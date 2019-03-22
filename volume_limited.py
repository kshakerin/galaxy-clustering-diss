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
(RA_mng, DEC_mng, z_mng, mr_mng, _) = md.MANGA_data_importer(data_directory + MaNGA_targets_filename)
# Set the RA/DEC window for galaxy observations
(RA_low, RA_high, DEC_low, DEC_high) = md.initialize_window(170, 20, 5, 5)
# Set bins
bin_num = 15
bin_select = np.logspace(np.log10(0.1), np.log10(50), bin_num, endpoint=True)


# Select the DATA, RANDOM, and MaNGA galaxies
dat = np.array([[RA_d[ii], DEC_d[ii], z_d[ii], mr[ii]] for ii in range(0, len(RA_d))])
ran = np.array([np.array([RA_r[ii], DEC_r[ii]]) for ii in range(0,len(RA_r))])
dat_mang = np.array([[RA_mng[ii], DEC_mng[ii], z_mng[ii], mr_mng[ii]] for ii in range(0, len(RA_mng))])

win = [RA_low, RA_high, DEC_low, DEC_high]
(data_3vec, data_cartesian, rand_3vec, rand_cartesian) = md.setup_data_and_randoms(dat, ran, win)
(manga_3vec, manga_cartesian, rand_mang_3vec, rand_mang_z_match) = md.setup_data_and_randoms(dat_mang, ran, win)


# Plot the volume limited samples
plt.figure(figsize=(15,7))
plt.scatter(dat[:,2], dat[:,3], marker="o", s=1, color="b", label="DR7")
plt.scatter(dat_mang[:,2], dat_mang[:,3], marker="o", s=1, color="r", label="MaNGA")
plt.title("Magnitude vs Redshift: DR7 & MaNGA")
plt.xlabel("Redshift")
plt.ylabel("Magnitude")
plt.ylim(-30,0)
plt.legend()

mag_lims = [-17,-18,-19]


# Do CCF Tests
(ccf_est, ccf_mang, ccf_dat, ccf_rand, v1, v2) = md.volume_limited_CCF(dat_mang, dat, ran, bin_select, win, mag_lims)
print(ccf_est)
# Plot a histgram of both the data redshifts and the random redshifts
plt.figure(figsize=(15,7))
for ii in range(0, len(mag_lims)):
	mmm = ccf_mang[ii]
	ddd = ccf_dat[ii]
	rrr = ccf_rand[ii]
	plt.hist(ddd[:,2], bins=50, density=1, label="DR7"   + str(mag_lims[ii]))
	plt.hist(rrr[:,2], bins=50, density=1, color="k", histtype="step", lw=1)
	plt.hist(mmm[:,2], bins=50, density=1, label="MaNGA" + str(mag_lims[ii]))
plt.title("Histograms of Data and Random Redshifts: CCF")
plt.xlabel("Redshift")
plt.ylabel("Number of Galaxies")
plt.legend()

# Plot the estimator
plt.figure(figsize=(15,7))
x_axis = (bin_select[1:] + bin_select[:-1])/2
for ii in range(0, len(mag_lims)):
	plt.plot(x_axis, ccf_est[ii], linewidth=2, markersize=12, label=str(mag_lims[ii]))
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title("Volume Limited CCF Calculations")
plt.xlabel("R [Mpc]")
plt.ylabel("Correlation Amplitude")
plt.legend()

# Plot the volume limited samples
plt.figure(figsize=(15,7))
plt.scatter(dat_mang[:,2], dat_mang[:,3], marker="o", s=1, color="b", label="All")
for jj in range(0, len(mag_lims)):
    sample = v1[jj]
    mag    = mag_lims[jj]
    plt.scatter(sample[:,2], sample[:,3], marker="v", s=1, label=str(mag))
plt.title("Magnitude vs Redshift: MaGNA")
plt.xlabel("Redshift")
plt.ylabel("Magnitude")
plt.ylim(-30,0)
plt.legend()

# Plot the volume limited samples
plt.figure(figsize=(15,7))
plt.scatter(dat[:,2], dat[:,3], marker="o", s=1, color="b", label="All")
for jj in range(0, len(mag_lims)):
    sample = v2[jj]
    mag    = mag_lims[jj]
    plt.scatter(sample[:,2], sample[:,3], marker="v", s=1, label=str(mag))
plt.title("Magnitude vs Redshift: DR7")
plt.xlabel("Redshift")
plt.ylabel("Magnitude")
plt.legend()


# Do ACF Tests
(est, dat_sph, _, ran_sph, _, _) = md.volume_limited_ACF(dat, ran, bin_select, win, mag_lims, 1)
print(est)
# Plot a histgram of both the data redshifts and the random redshifts
plt.figure(figsize=(15,7))
for ii in range(0, len(mag_lims)):
	ddd = dat_sph[ii]
	rrr = ran_sph[ii]
	plt.hist(ddd[:,2], bins=50, density=1, label=str(mag_lims[ii]))
	plt.hist(rrr[:,2], bins=50, density=1, color="k", histtype="step", lw=1)
plt.title("Histograms of Data and Random Redshifts: ACF")
plt.xlabel("Redshift")
plt.ylabel("Number of Galaxies")
plt.legend()

plt.figure(figsize=(15,7))
x_axis = (bin_select[1:] + bin_select[:-1])/2
for ii in range(0, len(mag_lims)):
	plt.plot(x_axis, est[ii], linewidth=2, markersize=12, label=str(mag_lims[ii]))
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title("Volume Limited ACF Calculations")
plt.xlabel("R [Mpc]")
plt.ylabel("Correlation Amplitude")
plt.legend()


"""
# Plot the volume limited samples
plt.figure(figsize=(15,7))
plt.scatter(mag_data[:,2], mag_data[:,3], marker="o", s=1, color="b", label="All")
for jj in range(0, len(subsamples)):
    sample = subsamples[jj]
    mag    = mag_lims[jj]
    plt.scatter(sample[:,2], sample[:,3], marker="o", s=1, label=str(mag))
plt.title("Volume Limited: Module")
plt.xlabel("Redshift")
plt.ylabel("Magnitude")
plt.legend()
"""

plt.show()
exit()

"""
# Plot the volume limited samples
plt.figure(figsize=(15,7))
plt.scatter(mag_data[:,2], mag_data[:,3], marker="o", s=1, color="b", label="All")
for jj in range(0, len(subsamples)):
    sample = subsamples[jj]
    mag    = mag_lims[jj]
    plt.scatter(sample[:,2], sample[:,3], marker="o", s=1, label=str(mag))
plt.title("Volume Limited: Module")
plt.xlabel("Redshift")
plt.ylabel("Magnitude")
plt.legend()
"""

"""
#OLD WORKING VERSION
# Make volume limited samples
# Arrange all data in ascending order by magnitude
mag_data = np.array(md.data_selector_prop(RA_d, DEC_d, z_d, mr, RA_low, RA_high, DEC_low, DEC_high))
mag_data = np.array([[RA_d[ii], DEC_d[ii], z_d[ii], mr[ii]] for ii in range(0, len(RA_d))])
sor_mag = mag_data[mag_data[:,3].argsort(kind="mergesort")]   # Sorts data in descending order on mag


# Setup mag-limited subsamples
subsamples = []
for lims in mag_lims:
    (vol_subsample, _, _) = md.volume_limited(dat, lims)
    subsamples.append(vol_subsample)

# Setup matching random catalogues for mag-limited subsamples
rand_subs = []
data_sph = []
data_subs = []
rand_3     = []
for subs in subsamples:
	(dat_sph, dat_c, ran_3, ran_c) = md.setup_data_and_randoms(subs, ran, win)
	data_subs.append(dat_c)
	data_sph.append(dat_sph)
	rand_3.append(ran_3)
	rand_subs.append(ran_c)

# Calculate the Estimator
estimators = []
for ii in range(0, len(data_subs)):
	estimators.append(md.ACF_estimator(data_subs[ii], rand_subs[ii], bin_select, 1))
	print(ii)

print(estimators)
# Plot a histgram of both the data redshifts and the random redshifts
plt.figure(figsize=(15,7))
for ii in range(0, len(mag_lims)):
	dat = data_sph[ii]
	ran = rand_3[ii]
	plt.hist(dat[:,2], bins=50, density=1, label=str(mag_lims[ii]))
	plt.hist(ran[:,2], bins=50, density=1, color="k", histtype="step", lw=1)
plt.title("Histograms of Data and Random Redshifts")
plt.xlabel("Redshift")
plt.ylabel("Number of Galaxies")
plt.legend()

# Plot the estimator
plt.figure(figsize=(15,7))
x_axis = (bin_select[1:] + bin_select[:-1])/2
for ii in range(0, len(mag_lims)):
	plt.plot(x_axis, estimators[ii], linewidth=2, markersize=12, label=str(mag_lims[ii]))
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title("Volume Limited ACF Calculations")
plt.xlabel("R [Mpc]")
plt.ylabel("Correlation Amplitude")
plt.legend()

# Plot the volume limited samples
plt.figure(figsize=(15,7))
plt.scatter(mag_data[:,2], mag_data[:,3], marker="o", s=1, color="b", label="All")
for jj in range(0, len(subsamples)):
    sample = subsamples[jj]
    mag    = mag_lims[jj]
    plt.scatter(sample[:,2], sample[:,3], marker="o", s=1, label=str(mag))
plt.title("Volume Limited")
plt.xlabel("Redshift")
plt.ylabel("Magnitude")
plt.legend()
plt.show()

exit()
"""

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
