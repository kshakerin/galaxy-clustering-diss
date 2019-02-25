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
MaNGA_targets_filename = "MaNGA_targets_extNSA_tiled_ancillary.fits"

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


###########################################
# Read in the MaNGA Target Catalogue data #
###########################################
hdu5              = fits.open(data_directory + MaNGA_targets_filename)
manga_target_data = hdu5[1].data
RA_mng            = manga_target_data.field(0)
DEC_mng           = manga_target_data.field(1)
z_mng             = manga_target_data.field(2)
abs_m_mng         = manga_target_data.field(3)
mr_mng            = [abs_m_d[i][2] for i in range(0,len(abs_m_mng))]

#pot.sky_positions_2([RA_mng,RA_d], [DEC_mng,DEC_d])
#pot.z_histogram_3(z_d, z_mng, False)
#plt.show()


# Window inforrmation
RA_start  = 205
DEC_start = 10
window    = 10
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

###########################
#  Select MaNGA Galaxies  #
###########################
(RA_3, DEC_3, z_3) = md.data_selector(RA_mng, DEC_mng, z_mng,
							   RA_low, RA_high, DEC_low, DEC_high)
manga_3vec         = [[RA_3[ii], DEC_3[ii], z_3[ii]] for ii in range(0, len(RA_3))]
manga_cartesian    = np.array([md.spherical_to_cartesian(ii[0], ii[1], ii[2])
						      for ii in manga_3vec])


plt.figure(figsize=(15,7))
plt.scatter(RA_2, DEC_2, marker="o", s=1, label="randoms")
plt.scatter(RA_1, DEC_1, marker="o", s=1, label="data")
plt.scatter(RA_3, DEC_3, marker="x", s=2, label="MaNGA")
plt.title("Spatial Distributions of Selections")
plt.xlabel("RA [deg]")
plt.ylabel("DEC [deg]")
plt.legend()
#plt.show()


# Split SDSS DR7 data on magnitude and test correlation






exit()


# Cross Correlation Tests
N_d     = len(data_cartesian)
N_r     = len(rand_cartesian)
D1D2    = md.improved_cross_correlate(manga_cartesian, data_cartesian, bin_select, 0)
D1R2    = md.improved_cross_correlate(manga_cartesian, rand_cartesian, bin_select, 0)
estim_1 = md.davis_peebles(D1D2, D1R2, N_d, N_r)
print(D1D2)
print(D1R2)
print(estim_1)

x_axis = (bin_select[1:] + bin_select[:-1])/2	# for plotting
plt.figure(figsize=(15,7))
plt.plot(x_axis, estim_1, linewidth=2, markersize=12,label=str("MaNGA Target CC"))
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title("Cross-Correlation Tests")
plt.xlabel("R [Mpc]")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

#import jackknife
exit()


# Plot galaxies and redshifts to check
#pot.sky_positions_3([RA_d, RA_1, RA_2], [DEC_d, DEC_1, DEC_2])
#pot.z_histogram_4(z_d, z_1, z_2)
#plt.show()

min_z    = min(z_2)
min_dist = model.comoving_distance(min_z).value
max_z    = max(z_2)
max_dist = model.comoving_distance(max_z).value
depth    = max_dist - min_dist
print("Depth of search area: ", depth, "[Mpc]")
print("Random z: \n", z_2[0])
print(str(window) + " Degree RA/DEC window size in Mpc for a random z: ")
print((window*60*model.kpc_comoving_per_arcmin(z_2[0]).value)/1000)

#(aa, bb) = md.jackknife_delete_d(data_cartesian, rand_cartesian, bin_select, 0, 1)
#print("Jackknife: ", aa)
#print("Norrmal: ", bb)

"""
# Example of estimators
x_axis = (bin_select[1:] + bin_select[:-1])/2
dps =  [101.76009770600446, 19.185507614328262, 12.228265638923522, 6.004294221072313,
		2.6867439019561163, 1.2044174194625166, 0.36107793122524745,
		0.08148608434727955, -0.0213561752731658]
dp  =  [176.85663573180517, 28.157273923891474, 12.376856733961702, 5.94468430204597,
		2.55024155685032, 1.1532684799749364, 0.35047375095024647,
		0.07731210801503274, -0.01278281433465378]
lz  =  [102.60455917596097, 19.800912630104566, 12.250481784015506, 5.987127157334432,
		2.6098464047464343, 1.1569092336851665, 0.34537354560865996,
		0.07373721324262417, -0.003987431975339906]

plt.figure(figsize=(15,7))
plt.plot(x_axis, dps, linewidth=2, markersize=12, label="Davis & Peebles Simple")
plt.plot(x_axis, dp, linewidth=2, markersize=12, label="Davis & Peebles")
plt.plot(x_axis, lz, linewidth=2, markersize=12, label="Landay & Szalay")
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title("Various Estimators at R=4*N")
plt.xlabel("R [Mpc]")
plt.ylabel("Correlation Amplitude")
plt.legend()
plt.show()
"""




est = []
for iii in range(1, 2):
	"""
	#DOWNSAMPLE
	# Downsampling Tests
	rand_cart_dwn = np.array([random.choice(rand_cartesian) for i
						  in range(0, iii*len(data_cartesian))])
	print("Downsampled random data", len(rand_cart_dwn))

	print("Number of data points:", len(data_cartesian))
	print("Number of rand points:", len(rand_cart_dwn))

	rand_len = len(rand_cart_dwn)

	#pair_counts_r  = md.auto_sort_pair_counter(rand_cart_dwn, bin_select, 0)
	#print("Pair counts AC rand:", pair_counts_r)

	"""
	rand_len = len(rand_cartesian)
	start = time.time()
	cc_bin_counts = md.improved_cross_correlate(data_cartesian, rand_cartesian,
											 bin_select, 0)
	end = time.time()
	tot_time = end-start
	print("Time taken: ", tot_time)

	"""
	# DO THE CC
	print
	cc_bin_counts = []
	for ii in range(0, len(bin_select)-1):
		rmin = bin_select[ii]
		rmax = bin_select[ii+1]
		cc_count = md.crude_cross_correlate(data_cartesian, rand_cart_dwn, rmin, rmax)
		if (cc_count == 0):
			cc_count = 1
		cc_bin_counts.append(cc_count)
	"""

	print("CC bin counts:", cc_bin_counts)


	# Calculate correlation function estimator
	estimator = []
	norm      = (rand_len*(rand_len-1))/(data_len*(data_len-1))
	for ii in range(0, len(pair_counts_d)):
		aa = pair_counts_d[ii]/cc_bin_counts[ii]
		bb = norm**aa # MINUS 1 Removed in order to make plotting look nicer
		estimator.append(bb)
	print("Davis and Peebles simple estimator: " + "\n", estimator)
	print("Data window:")
	print(str(RA_start) + " to " + str(RA_high) + " RA [deg]")
	print(str(DEC_start) + " to " + str(DEC_high) + " DEC [deg]")

	est.append(estimator)

cl = ["r", "g", "b", "y"]
plt.figure(figsize=(15,7))
x_axis = (bin_select[1:] + bin_select[:-1])/2
for jjj in range(0, len(est)):
	color = cl[jjj] + "o--"
	plt.plot(x_axis, est[jjj], color, linewidth=2, markersize=12, label=str(jjj+1))
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title("SDSS DR7 Correlation Estimator")
plt.xlabel("R [Mpc]")
plt.ylabel("Correlation Amplitude (Davis and Peebles)")
plt.legend()
plt.show()






"""

# sort the data for quicker calculations
sorted_set1    = data_cartesian[data_cartesian[:,0].argsort()][::-1]
sorted_set2    = rand_cartesian[rand_cartesian[:,0].argsort()][::-1]
#print("Set1: ")
#print(sorted_set1[0:3])
#print("Set2: ")
#print(sorted_set2[0:3])

dmax = 1
for ii in range(0, len(sorted_set1)):
	v0 = sorted_set1[ii]
	count = ii
	indices = np.where(abs(sorted_set2[:,0]-v0[0]) < dmax)
	if len(indices[0]) == 0:
		continue
	print(indices)
	print(indices[0][-1])
	print(count)
	print(sorted_set2[indices])

	new_index = indices[0][-1]
	top_half = sorted_set2[0:new_index]
	bottom_half = sorted_set2[new_index:-1]

exit()
"""





exit()
"""
galaxy1 = data_3vec[0]
galaxy2 = data_3vec[1]
print(model.comoving_distance(galaxy1[2]).value)
print(model.__doc__)
print("H0: ", model.H(0))

print("Galaxy 1:", galaxy1)
print("Galaxy 2:", galaxy2)
# Calculate distance using sperical
d_sphere = distance_spherical(galaxy1, galaxy2)
print("Distance calculated using spherical: ", d_sphere)
#Calculate distance using cartesian
vec1 = spherical_to_cartesian(galaxy1[0],galaxy1[1],galaxy1[2])
vec2 = spherical_to_cartesian(galaxy2[0],galaxy2[1],galaxy2[2])
d_cart = distance_cartesian(vec1, vec2)
print("Distance calculated using cartesian: ", d_cart)
"""


### autocorrelation tests ###
print("autocorrelation tests")
# Set bins
bin_select     = np.logspace(np.log10(0.1), np.log10(50), 10, endpoint=True)
# Count pairs for the data set
data_cartesian = np.array([spherical_to_cartesian(ii[0], ii[1], ii[2]) for ii in data_3vec])
pair_counts_d  = auto_sort_pair_counter(data_cartesian, bin_select, 0)
print("Data set pair counts:", pair_counts_d)
# Count pairs for the random set
rand_cartesian = np.array([spherical_to_cartesian(ii[0], ii[1], ii[2]) for ii in rand_3vec])
pair_counts_r  = auto_sort_pair_counter(rand_cartesian, bin_select, 0)
print("Random set pair counts:", pair_counts_r)

# Calculate correlation function estimator
estimator = []
norm      = rand_len/data_len
for ii in range(0, len(pair_counts_d)):
	aa = pair_counts_d[ii]/pair_counts_r[ii]
	bb = norm*aa # MINUS 1 Removed in order to make plotting look nicer
	estimator.append(bb)
print("Davis and Peebles simple estimator: " + "\n", estimator)
print("Data window:")
print(str(RA_start) + " to " + str(RA_high) + " RA [deg]")
print(str(DEC_start) + " to " + str(DEC_high) + " DEC [deg]")


plt.figure(figsize=(15,7))
x_axis = (bin_select[1:] + bin_select[:-1])/2
plt.plot(x_axis, estimator, 'go--', linewidth=2, markersize=12)
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title("SDSS DR7 Correlation Estimator")
plt.xlabel("R [Mpc]")
plt.ylabel("Correlation Amplitude (Davis and Peebles)")
plt.show()



exit()

#print(cart_3vec[0:3])
#sort on the cartesian x-axis
sorted_3vec = cart_3vec[cart_3vec[:,0].argsort()][::-1]
sorted_x = cart_3vec[cart_3vec[:,0].argsort()][::-1]
sorted_y = cart_3vec[cart_3vec[:,1].argsort()][::-1]
sorted_z = cart_3vec[cart_3vec[:,2].argsort()][::-1]
#print(len(sorted_3vec))
#print(len(cart_3vec))
#print(sorted_3vec[0:3])
rmin = 1.
rmax = 10.
start = time.time()
counter = 0
for ii in range(0, len(sorted_3vec)):
	v0 = sorted_3vec[ii]
	for jj in range(ii+1, len(sorted_3vec)):
		v1 = sorted_3vec[jj]
		#print(v0)
		#print(v1)
		#print(v0[0]-v1[0])
		if (v0[0] - v1[0]) > rmax:
			break
		else:
			dist = distance_cartesian(v0, v1)
			#print(dist)
			if (dist >= rmin) & (dist <= rmax):
				counter += 1
print("Pair Counts:", counter)
stop = time.time()
tot_time = stop-start
print("Time Taken: ", tot_time)

exit()
start = time.time()
count = crude_pair_counter_v2(cart_3vec, cart_3vec, 1, 10)
print(count)
stop = time.time()
tot_time = stop-start
print("Old version time: ", tot_time)

exit()
"""
# Select random z values from the DR7 data catelogue of z
# Record time taken
start = time.time()
z_r   = random_z_assign(RA_r, DEC_r, z_d)
end   = time.time()
tt    = end-start
#print("Total time taken for random redshift assignment: ", tt)
# Print number of random redshifts as a check
#print("Number of random reshifts: ", len(z_r))
# Select a subsample for correlation function testing
RA_low   = 181
RA_high  = 182
DEC_low  = 20
DEC_high = 21


# Select Galaxies
(RA_1, DEC_1, z_1) = data_selector(RA_d, DEC_d, z_d,
								   RA_low, RA_high, DEC_low, DEC_high)
print("Number of galaxies selected:", len(RA_1))

#Randomly Select galaxies
(RA_2, DEC_2, z_2) = random_selector(RA_r, DEC_r, z_1,
									 RA_low, RA_high, DEC_low, DEC_high)
print("Number of random galaxies selected:", len(RA_2))

# Comoving distances
# initial comoving distance attempt
# using spherical coordinates

# Use RA_1, DEC_1, and z_1
data_3vec   = [[RA_1[i], DEC_1[i], z_1[i]] for i in range(0, len(RA_1))]
random_3vec = [[RA_2[i], DEC_2[i], z_2[i]] for i in range(0, len(RA_2))]

close_RA = []
close_DEC = []
close_z = []

dr_l = .1 # Mpc
dr_h = 1.

# do it for both the actual data
counter1 = 0
for i in range(0, len(RA_1)):
	v0 = data_3vec[i]
	for j in range(i+1, len(RA_1)):
		v1 = data_3vec[j]
		#print(v1)
		cmoved = distance_spherical(v0,v1)
		#print(cmoved)
		if (cmoved >= dr_l) & (cmoved <= dr_h):
			counter1 += 1
			#close_RA.append(v1[0])
			#close_DEC.append(v1[1])

print("Number of data pairs in bin " + str(dr_l) + " to " + str(dr_h) +
	  " = ", counter1)
corr1 = counter1/(len(data_3vec)**2)
print("Corrected for # of data points: ", corr1)

#do it for the random data
counter2 = 0
for i in range(0, len(RA_1)):
	v0 = data_3vec[i]
	for j in range(i+1, len(RA_2)):
		v1 = random_3vec[j]
		#print(v1)
		cmoved = distance_spherical(v0,v1)
		#print(cmoved)
		if (cmoved >= dr_l) & (cmoved <= dr_h):
			counter2 += 1
			#close_RA.append(v1[0])
			#close_DEC.append(v1[1])

print("Number of random pairs in bin " + str(dr_l) + " to " + str(dr_h) +
	  " = ", counter2)
corr2 = counter2/(len(random_3vec)**2)
print("Corrected for # of data points: ", corr2)

dp_estim = ((counter1*len(random_3vec))/(counter2*len(data_3vec))) - 1.
print("Davis and Peebles Estimator: ", dp_estim)
"""


# Optimizing pair counting
print("BEGIN TIME TRIALS")
RA_start  = 180
DEC_start = 25
window_size = [1,2,3,4,5,6,7,8,9,10] # degrees in RA and DEC
for i in window_size[0:2]:
	# Set the RA/DEC window
	RA_low   = RA_start
	RA_high  = RA_start + i
	DEC_low  = DEC_start
	DEC_high = DEC_start + i
	print("Window size:" + str(i) + " by " + str(i) + " deg")
	# Select the data points in the window
	(RA_1, DEC_1, z_1) = data_selector(RA_d, DEC_d, z_d,
								   RA_low, RA_high, DEC_low, DEC_high)
	(RA_2, DEC_2, z_2) = random_selector(RA_r, DEC_r, z_d,
								   RA_low, RA_high, DEC_low, DEC_high)
	print("Number of galaxies selected:", len(RA_1))
	print("Number of random selected:", len(RA_2))
	print(len(z_2))
	data_3vec_spher = [[RA_1[i], DEC_1[i], z_1[i]] for i in range(0, len(RA_1))]
	# Convert redshift into comoving coordinates
	z_com_1 = model.comoving_distance(z_1).value
	z_com_2 = model.comoving_distance(z_2).value
	print("z_1", len(z_com_1))
	print("z_2", len(z_com_2))
	print("Redshifts converted to comiving coordinates")
	# Set up the data in vectorized form
	data_3vec   = [[RA_1[i], DEC_1[i], z_com_1[i]] for i in range(0, len(RA_1))]
	d_len       = len(data_3vec)
	random_3vec = [[RA_2[i], DEC_2[i], z_com_2[i]] for i in range(0, len(RA_2))]
	r_len       = len(random_3vec)
	print("RA, DEC, and distance packaged ino vector form")
	# Convert to cartesian coordiantes
	data_cart = [spherical_to_cartesian(ii[0], ii[1], ii[2]) for ii in data_3vec]
	rand_cart = [spherical_to_cartesian(jj[0], jj[1], jj[2]) for jj in random_3vec]
	print("Spherical converted to cartesian")
	# Begin the pair counting for DD and DR pairs
	start       = time.time()
	DD_count_1    = crude_pair_counter(data_3vec_spher, data_3vec_spher, .1, 1)
	end         = time.time()
	tot_time    = end - start
	print("sphere time", tot_time)
	start       = time.time()
	DD_count_2 = crude_pair_counter_v2(data_3vec, data_3vec, .1, 1)
	end         = time.time()
	tot_time    = end - start
	print("cartesian time", tot_time)
	#DR_count    = crude_pair_counter(data_3vec, random_3vec, .1, 10)
	print("v1: ", DD_count_1)
	print("v2: ", DD_count_2)
	#print("Number of pair counts (DD): ", DD_count)
	#print("Number of data random counts(DR): ", DR_count)
	#print("Time taken to count pairs in a window of size: "
	#        + str(i) + " by " + str(i) + " degrees:", tot_time)
	#DP_estim    = davis_peebles(DD_count, DR_count, d_len, r_len)
	#print("Davis and Peebles estimator ~ ", DP_estim)
#RESULT - BAD/SLOW


#############################################################


# End
