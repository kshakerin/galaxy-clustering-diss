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

"""
#DR7
tab = fits.open(data_directory + DR7_data_filename)
tab.info()
aa = tab[0].header
print(aa)
#MaNGA
tab2 = fits.open(data_directory + MaNGA_targets_filename)
tab2.info()
aa2 = tab2[0].header
print(aa2)
"""



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
(RA_low, RA_high, DEC_low, DEC_high) = md.initialize_window(180, 10, 10, 10)
win 								 = [RA_low, RA_high, DEC_low, DEC_high]
# Set bins
bin_num    = 10
bin_select = np.logspace(np.log10(0.1), np.log10(50), bin_num, endpoint=True)
x_axis     = (bin_select[1:] + bin_select[:-1])/2



"""
aa = [41.71504121947426, 42.68035288545107, 19.130948970219958, 12.229432527760007, 6.568575038092856, 3.311935502665132, 1.6224977960681586, 0.9177184315133471, 0.9253696736816885]
bb = [104.77057825774578, 57.07012139640944, 21.616868997074214, 12.434165764362422, 6.627143433729365, 3.2880444103907482, 1.6714809105917579, 0.9522857834817009, 0.9552357946832153]
cc = [112.90677658526468, 131.89123934947546, 37.4620284573621, 25.574821071562205, 10.606923368878183, 4.735511758804893, 2.3173446071122585, 1.1964851800616487, 1.0022688305928527]
"""
aa = [149.7958424405897, 56.494067017482365, 24.173875255668584, 13.250631653409183, 6.348474267461999, 3.2258127221240436, 1.8090823395587403, 1.1118612096754688, 1.047501897978952]
bb = [226.28648715683082, 80.13471957401293, 30.838172413254618, 16.592848284735464, 7.826467282177964, 3.8306579660787126, 2.0862729131617126, 1.2328999902263353, 1.0487994585313585]
cc = [418.60582244338, 90.78877365948938, 40.67318099608911, 26.496896697994224, 11.899635232020183, 5.591997486762397, 2.807601571900217, 1.36956892272558, 1.0767630146909388]
estimators = [aa,bb,cc]


limits = [-15,-17,-19]

plt.figure(figsize=(15,7))
for ii in range(0, len(limits)):
	plt.plot(x_axis, estimators[ii], linewidth=2,
									 markersize=12,
									 label=str(limits[ii]) + " Abs Mag")
plt.xlabel("R [Mpc]")
plt.ylabel("Correlation Amplitude")
plt.title("MaNGA Volume Limited CCF with Dr7")
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.legend()

plt.show()

exit()



# Begin Test of CCF using various MaNGA magnitude bins
# Step 1: Select the DATA, MaNGA, and the RANDOM galaxies
dat_mang = np.array([[RA_mng[ii], DEC_mng[ii], z_mng[ii], mr_mng[ii], corr[ii]] for ii in range(0, len(RA_mng))])
"""
plt.figure(figsize=(7,7))
plt.scatter(dat_mang[:,2], dat_mang[:,3], color="b", s=1, label="MaNGA")
plt.xlabel("Redshift")
plt.ylabel("Magnitude")
plt.ylim(-30,0)
plt.xlim(0,.15)
plt.title("Luminosity Bins: DR7")
plt.legend()
plt.show()
exit()
"""
dat      = np.array([[RA_d[ii], DEC_d[ii], z_d[ii], mr[ii]] for ii in range(0, len(RA_d))])
ran      = np.array([[RA_r[ii], DEC_r[ii]] for ii in range(0,len(RA_r))])
# Now remove the MaNGA targets from the data catalogue
dat0	 = md.coordinate_match(dat_mang, dat)
# Setup the MaNGA data in the specific window
(m_sp, m_c, _, _) = md.setup_data_and_randoms(dat_mang, ran, win)
# Step 2: Setup the Magnitude cuts for the MaNGA data
limits = [-15,-17,-19]
subs   = []
for limit in limits:
	(mag_sub, _, _)       = md.volume_limited(dat_mang, limit)
	(sub_sp, sub_c, _, _) = md.setup_data_and_randoms(mag_sub, ran, win)
	subs.append(sub_c)
print("Mag Cuts Finished")

print(m_c[0:3])
# Set the redshift limits
zmin			  = 0.01
zmax 			  = max(m_sp[:,2])
print("z min and max: ", zmin, zmax)
dat1			  = dat0[np.where((dat0[:,2] > zmin) & (dat0[:,2] < zmax))]
# Setup the data and randoms in specific window
(d_sp, d_c, r_sp, r_c) = md.setup_data_and_randoms(dat1, ran, win)
print("MaNGA data set size: ", len(m_c))
print("DR7 data set size: ", len(d_c))
print("DR7 randoms set size: ", len(r_c))


print("Begin Estimator Calculations")

# Step 3: Calculate the CCF for each magnitude bin
estimators = []
for sub in subs:
	estimator = md.CCF_estimator(sub, d_c, r_c, bin_select)
	estimators.append(estimator)
	print(estimator)

plt.figure(figsize=(15,7))
for ii in range(0, len(limits)):
	plt.plot(x_axis, estimators[ii], linewidth=2,
									 markersize=12,
									 label=str(limits[ii]) + " Abs Mag")
plt.xlabel("R [Mpc]")
plt.ylabel("Correlation Amplitude")
plt.title("MaNGA Volume Limited CCF with Dr7")
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.legend()

plt.show()

exit()



# Select the DATA, MaNGA, and the RANDOM galaxies
dat      = np.array([[RA_d[ii], DEC_d[ii], z_d[ii], mr[ii]] for ii in range(0, len(RA_d))])
ran      = np.array([[RA_r[ii], DEC_r[ii]] for ii in range(0,len(RA_r))])
dat_mang = np.array([[RA_mng[ii], DEC_mng[ii], z_mng[ii], mr_mng[ii], corr[ii]] for ii in range(0, len(RA_mng))])
(data_3vec, data_cartesian, rand_3vec, rand_cartesian)           = md.setup_data_and_randoms(dat, ran, win)
(manga_3vec, manga_cartesian, rand_mang_3vec, rand_mang_z_match) = md.setup_data_and_randoms(dat_mang, ran, win)


print("Begin Luminosity Bin Splitting")
bins   = [-17,-18,-19,-20,-21,-22,-23]
subs   = []
z_lims = []
for ii in range(0, len(bins)-1):
	lims = [bins[ii], bins[ii+1]]
	(vol_subsample, zmin, zmax, limits) = md.volume_limited_bins(dat, lims)
	subs.append(vol_subsample)
	z_lims.append([zmin, zmax])

print("Begin ACF Estimations")
acf_estimates = []
count = 0
for jj in subs:
	count += 1
	print(count)
	(d_sp, d_c, r_sp, r_c) = md.setup_data_and_randoms(jj, ran, win)
	est = md.ACF_estimator(d_c, r_c, bin_select, 1)
	acf_estimates.append(est)

plt.figure(figsize=(15,7))
for ii in range(0, len(bins)-1):
	est = acf_estimates[ii]
	plt.plot(x_axis, est, linewidth=2, markersize=12, label=str(bins[ii]) + " to " + str(bins[ii+1]))
plt.xlabel("Redshift")
plt.ylabel("Magnitude")
plt.title("ACF Estimators for Various Luminosity Bins")
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.legend()


plt.figure(figsize=(7,7))
plt.scatter(dat[:,2], dat[:,3], color="b", s=1, label="SDSS")
for ii in range(0, len(bins)-1):
	data = subs[ii]
	plt.scatter(data[:,2], data[:,3], s=1, label=str(bins[ii]) + " to " + str(bins[ii+1]))
	"""
	xx    = z_lims[ii][0]
	xdelt = z_lims[ii][1] - z_lims[ii][0]
	yy    = bins[ii]
	ydelt = bins[ii+1] - bins[ii]
	p = patch.Rectangle((xx,yy),xdelt,ydelt,linewidth=1,edgecolor='r',facecolor='none')
	ax.add_patch(p)
	"""
#plt.scatter(dat_mang[:,2], dat_mang[:,3], color="r", s=1, label="MaNGA Petrosian")
#plt.scatter(dat_mang[:,2], dat_mang[:,4], color="blue", s=1, label="MaNGA Serisic")
plt.xlabel("Redshift")
plt.ylabel("Magnitude")
plt.ylim(-23,-17)
plt.xlim(0,.25)
plt.gca().invert_yaxis()
plt.title("Luminosity Bins: DR7")
plt.legend()
plt.show()

exit()

# Remove the matching galaxies from the main data catalogue
(new_data, matches) = md.remove_from_catalogues(manga_3vec, data_3vec)
new_data_cartesian  = np.array([md.spherical_to_cartesian(ii[0], ii[1], ii[2])
						        for ii in new_data])
print("DR7 set size: ", len(data_cartesian))
print("MaNGA set size: ", len(manga_cartesian))
print("Removed data set size:", len(new_data_cartesian))
print("Change: ", len(data_cartesian)-len(new_data_cartesian))


new_data_spherical = md.coordinate_match(manga_3vec, data_3vec)
Nd_dr7 = len(new_data_spherical)
print("New DR7 Nd (after removing overlap with MaNGA sample): ", Nd_dr7)

(_, new_data_cartesian, _, new_data_rands) = md.setup_data_and_randoms(new_data_spherical, ran, win)


# Find the ratio of DR7/MaNGA as a function of redshift
z_bins = np.linspace(0, .14, num=150, endpoint=True)
ratio  = []
for ii in range(0, len(z_bins)-1):
	z_min = z_bins[ii]
	z_max = z_bins[ii+1]
	data_z = len(z_d[np.where((z_d > z_min) & (z_d < z_max))])
	print(data_z)
	mang_z = len(z_mng[np.where((z_mng > z_min) & (z_mng < z_max))])
	print(mang_z)
	if mang_z == 0:
		rat = 0
	else:
		rat    = data_z/mang_z
	print(rat)
	ratio.append(rat)

alt_x_axis = (z_bins[1:] + z_bins[:-1])/2
plt.figure()
plt.plot(alt_x_axis, ratio)
plt.title("DR7/MaNGA as a Funciton of z")
plt.show()

exit()



"""
# Limit the whole sample out to some redshift
z_lim              = model.comoving_distance(0.04).value
manga_cartesian    = manga_cartesian[np.where(manga_cartesian[:,2] < z_lim)]
new_data_cartesian = new_data_cartesian[np.where(new_data_cartesian[:,2] < z_lim)]
new_data_rands     = new_data_rands[np.where(new_data_rands[:,2] < z_lim)]
print("Length of Data:", len(data_3vec))
print("Length of MaNGA:", len(manga_3vec))
print("Number of lefovers:", Nd_dr7)
print("number of randoms:", len(new_data_rands))
"""

#print(indexes)
print("Length of DR7:", len(data_3vec))
print("Length of MaNGA:", len(manga_3vec))
print("Number of lefovers:", Nd_dr7)
print("number of randoms:", len(new_data_rands))

CCF       = md.CCF_estimator(manga_cartesian, new_data_cartesian, new_data_rands, bin_select)
ACF_manga = md.ACF_estimator(manga_cartesian, rand_mang_z_match, bin_select, 1)
ACF_data  = md.ACF_estimator(new_data_cartesian, new_data_rands, bin_select, 1)

print("CCF \n", CCF)
print("ACF MaNGA \n", ACF_manga)
print("ACF Data \n", ACF_data)


# NOTE: Probably Double Counting Galaxy pairs
x_axis = (bin_select[1:] + bin_select[:-1])/2	# for plotting
plt.figure(figsize=(15,7))
plt.plot(x_axis, CCF, linewidth=2, markersize=12, label="CCF")
plt.plot(x_axis, ACF_manga, linewidth=2, markersize=12, label="ACF_MaNGA")
plt.plot(x_axis, ACF_data, linewidth=2, markersize=12, label="ACF_DR7")
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title("Cross-Correlation Tests (MaNGA Targets with DR7)")
plt.xlabel("R [Mpc]")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

"""
plt.figure(figsize=(15,7))
plt.scatter(RA_d, DEC_d, marker='o', s=1)
plt.title("Galaxy Catelogue Sky Positions")
plt.xlabel("RA [deg]")
plt.ylabel("DEC [deg]")
"""

plt.show()

exit()
#############################################################


# End
