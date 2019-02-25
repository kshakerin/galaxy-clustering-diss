import numpy as np
import math as mt
import pandas as pd
import os
from astropy.io import fits
from astropy.io import ascii
from astropy.cosmology import Planck15 as model
#from sklearn.neighbors import BallTree
import random
import time

#############################################################
#-----------------------------------------------------------#
#-----------------------------------------------------------#
#                        Functions                          #
#-----------------------------------------------------------#
#-----------------------------------------------------------#
#############################################################

#-----------------------------------------------------------#
def random_z_assign(RA_random, DEC_random, z_select):
#-----------------------------------------------------------#
# Randomly assign redshifts to the random data set using
# redshifts from the selection of actual data.
#-----------------------------------------------------------#
	z_out = np.array([random.choice(z_select) for i
					  in range(0, len(RA_random))])
	return z_out

#-----------------------------------------------------------#
def data_selector(RA, DEC, Z, RA_l, RA_h, DEC_l, DEC_h):
#-----------------------------------------------------------#
# Selects some data in a given RA/DEC range and outputs the
# RA/DEC pairs with the corresponding redshifts.
#-----------------------------------------------------------#
	# Slice data on RA
	ind1   = np.where((RA > RA_l) & (RA < RA_h))
	RA_v1  = RA[ind1]
	DEC_v1 = DEC[ind1]
	Z_v1   = Z[ind1]
	# Slice data on DEC
	ind2   = np.where((DEC_v1 > DEC_l) & (DEC_v1 < DEC_h))
	RA_vf  = RA_v1[ind2]
	DEC_vf = DEC_v1[ind2]
	Z_vf   = Z_v1[ind2]
	# Output in the format (RA, DEC, Z)
	return RA_vf, DEC_vf, Z_vf

#-----------------------------------------------------------#
def random_selector(RA, DEC, Z_real, RA_l, RA_h, DEC_l, DEC_h):
#-----------------------------------------------------------#
# Selects data from the random set in a given RA/DEC range
# and outputs the RA/DEC pairs with the corresponding
# redshifts.
#-----------------------------------------------------------#
	# Slice data on RA
	ind1   = np.where((RA > RA_l) & (RA < RA_h))
	RA_v1  = RA[ind1]
	DEC_v1 = DEC[ind1]
	# Slice data on DEC
	ind2   = np.where((DEC_v1 > DEC_l) & (DEC_v1 < DEC_h))
	RA_vf  = RA_v1[ind2]
	DEC_vf = DEC_v1[ind2]
	# Randomly assign the Redshifts
	Z_vf   = random_z_assign(RA_vf, DEC_vf, Z_real)
	# Output in the format (RA, DEC, Z)
	return RA_vf, DEC_vf, Z_vf

def distance_spherical(v_0, v_1):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
	d_1 = model.comoving_distance(v_0[2]).value
	d_2 = model.comoving_distance(v_1[2]).value
	aa = np.sin(np.deg2rad(v_0[0]))*np.sin(np.deg2rad(v_1[0]))
	bb = aa*np.cos(np.deg2rad(90. - v_0[1]) - np.deg2rad(90. - v_1[1]))
	cc = bb + (np.cos(np.deg2rad(v_0[0]))*np.cos(np.deg2rad(v_1[0])))
	dd = 2.*d_1*d_2*cc
	ee = d_1*d_1 + d_2*d_2
	return np.sqrt(ee - dd)

#-----------------------------------------------------------#
def crude_pair_counter(DSET1, DSET2, r1, r2):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
	count = 0  # Set count to zero
	for ii in range(0, len(DSET1)): # Loop over each point
		v0 = DSET1[ii]
		for jj in range(ii + 1, len(DSET2)):
			# Compare to all points except the ones before - don't double count
			v1 = DSET2[jj]
			dist = distance_spherical(v0,v1)   # find distance btwn pair
			if (dist >= r1) & (dist <= r2):
				count += 1 # increment pair count
	return count


#-----------------------------------------------------------#
def spherical_to_cartesian(theta, phi, z_val):
#-----------------------------------------------------------#
# Converts sphericaal coordiantes to cartesian coordiantes.
# Input:
#   theta: polar angle - float, degrees
#     phi: azimuthal angle - float, degrees
#   z_val: redshift - float
# Output:
#   numpy array: [x, y, z]
#-----------------------------------------------------------#
	radial = model.comoving_distance(z_val).value
	phi    = 90. - phi
	xx     = radial*np.cos(np.deg2rad(theta))*np.sin(np.deg2rad(phi))
	yy     = radial*np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(phi))
	zz     = radial*np.cos(np.deg2rad(phi))
	return [xx, yy, zz]

#-----------------------------------------------------------#
def distance_cartesian(vec1, vec2):
#-----------------------------------------------------------#
# Determines the distance between 2 points in cartesian
# coordiantes.
# Input:
#   vec1: cartesian 3-vector, float
#   vec2: cartesian 3-vector, float
# Output:
#   distance: distance between the two input vectors
#-----------------------------------------------------------#
	xx   = (vec1[0] - vec2[0])**2
	yy   = (vec1[1] - vec2[1])**2
	zz   = (vec1[2] - vec2[2])**2
	dist = np.sqrt(xx + yy + zz)
	return dist

#-----------------------------------------------------------#
def crude_pair_counter_v2(DSET1, DSET2, r1, r2):
#-----------------------------------------------------------#
# Uses the cartesian system
#-----------------------------------------------------------#
	count = 0  # Set count to zero
	for ii in range(0, len(DSET1)): # Loop over each point
		v0 = DSET1[ii]
		for jj in range(ii + 1, len(DSET2)):
			# Compare to all points except the ones before - don't double count
			v1 = DSET2[jj]
			dist = distance_cartesian(v0,v1)   # find distance btwn pair
			if (dist >= r1) & (dist <= r2):
				count += 1 # increment pair count
	return count

#-----------------------------------------------------------#
def auto_sort_pair_counter(DSET1, bins, key):
#-----------------------------------------------------------#
# Uses the cartesian system.
# Calculates the autocorrelatin of a set of data for a
# a given set of bins (r - r+dr).
# Sorts the data on the given "key" input:
#   0 --> x-axis
#   1 --> y-axis
#   2 --> z-axis
#-----------------------------------------------------------#
	#print("Number of Galaxies: ", len(DSET1))
	# sort the data for quicker calculations
	sorted_set1    = DSET1[DSET1[:,key].argsort()][::-1]
	# Get numbr of bins
	bl             = len(bins) - 1
	# Initialize output array
	bin_counts_out = []
	for ii in range(0, bl):
		# set the bin min and max
		rmin    = bins[ii]
		rmax    = bins[ii+1]
		start   = time.time() # Check speed of calculation
		counter = 0
		for jj in range(0, len(sorted_set1)):
			v0 = sorted_set1[jj]
			for kk in range(jj+1, len(sorted_set1)):
				v1 = sorted_set1[kk]
				if (v0[0] - v1[0]) > rmax:
					break
				else:
					dist = distance_cartesian(v0, v1)
					#print(dist)
					if (dist >= rmin) & (dist <= rmax):
						counter += 1    # Increment counter for each pair in bin
		if (counter == 0):
			counter = 1
		bin_counts_out.append(counter)  # store number of pair counts in each bin
		#print("Pair Counts in bin "+ str(rmin) +
		#      " to " + str(rmax) + " Mpc:", counter)
		#stop     = time.time()
		#tot_time = stop-start
		#print("Time Taken: ", tot_time)

	return bin_counts_out


#-----------------------------------------------------------#
def crude_cross_correlate(DSET1, DSET2, r1, r2):
#-----------------------------------------------------------#
# Uses the cartesian system.
# Finds the cross correlation between DSET1 and DSET2
#-----------------------------------------------------------#
	count = 0  # Set count to zero
	for ii in range(0, len(DSET1)): # Loop over each point
		v0 = DSET1[ii]
		for jj in range(0, len(DSET2)):
			# Compare to all points except the ones before - don't double count
			v1 = DSET2[jj]
			dist = distance_cartesian(v0,v1)   # find distance btwn pair
			if (dist > r1) & (dist < r2):
				count += 1 # increment pair count
	return count

#-----------------------------------------------------------#
def best_cross_correlate(DSET1, DSET2, bins):
#-----------------------------------------------------------#
# DOES NOT WORK!!!!!!!!!!!!!!!!!!
# Uses the cartesian system.
# Finds the cross correlation between DSET1 and DSET2
#-----------------------------------------------------------#
	bl             = len(bins) - 1
	# Initialize output array
	bin_counts_out = []
	for xx in range(0, bl):
		count = 0 # initialize counter
		# set the bin min and max
		rmin    = bins[xx]
		rmax    = bins[xx+1]
		for ii in range(0, len(DSET1)): # Loop over each point
			v0 = DSET1[ii]
			indices1 = np.where((distance_cartesian(v0, DSET2[:,0:2]) > rmin)
							& (distance_cartesian(v0, DSET2[:,0:2]) < rmax))
			#print(indices1)
			#print(indices2)
			count += len(indices1)
		bin_counts_out.append(count)
	return bin_counts_out

#-----------------------------------------------------------#
def improved_cross_correlate(DSET1, DSET2, bins, key):
#-----------------------------------------------------------#
# Uses the cartesian system.
# Finds the cross correlation between DSET1 and DSET2
#-----------------------------------------------------------#
	#print("Number of DSET1 Galaxies: ", len(DSET1))
	#print("Number of DSET2 Galaxies: ", len(DSET2))
	# sort the data for quicker calculations
	#sorted_set1    = DSET1[DSET1[:,key].argsort()][::-1]
	#sorted_set2    = DSET2[DSET2[:,key].argsort()][::-1]
	# Get numbr of bins
	bl             = len(bins) - 1
	# Initialize output array
	bin_counts_out = []
	for xx in range(0, bl):
		counter = 0 # initialize counter
		# set the bin min and max
		rmin    = bins[xx]
		rmax    = bins[xx+1]
		# Find entry point
		iic = 0 # initial index count
		for v0 in DSET1:
			selec = DSET2[np.where((abs(DSET2[:,0]-v0[0]) < rmax))]
			for ii in selec:
				dist = distance_cartesian(v0, ii)
				#print(dist)
				if (dist > rmin) & (dist < rmax):
					counter += 1    # Increment counter for each pair in bin
		if counter == 0:
			counter = 1
		bin_counts_out.append(counter)

	return bin_counts_out

#-----------------------------------------------------------#
def davis_peebles_simple(DD, RR, nd, nr):
#-----------------------------------------------------------#
# Simple davis_peebles estimator using DD and RR
#-----------------------------------------------------------#
	nd = float(nd)
	nr = float(nr)
	norm  = (nr*(nr-1.))/(nd*(nd-1.))
	estim = []
	for ii in range(0, len(DD)):
		aa = (norm*(float(DD[ii])/float(RR[ii]))) - 1.
		estim.append(aa)
	return estim

#-----------------------------------------------------------#
def davis_peebles(DD, DR, nd, nr):
#-----------------------------------------------------------#
# Davis and Peebles estimator using DD and DR
#-----------------------------------------------------------#
	nd = float(nd)
	nr = float(nr)
	norm  = (2.*nr)/(nd-1.)
	estim = []
	for ii in range(0, len(DD)):
		aa = norm*(float(DD[ii])/float(DR[ii])) - 1.
		estim.append(aa)
	return estim

#-----------------------------------------------------------#
def landay_szalay(DD, RR, DR, nd, nr):
#-----------------------------------------------------------#
# Landay and Szalay estimator using DD, RR, and DR.
# Most optimized estimator, espiecially for large R
#-----------------------------------------------------------#
	nd = float(nd)
	nr = float(nr)
	norm1 = (nr*(nr-1.))/(nd*(nd-1.))
	norm2 = (nr-1.)/nd
	estim = []
	for ii in range(0, len(DD)):
		aa = norm1*(float(DD[ii])/float(RR[ii]))
		bb = norm2*(float(DR[ii])/float(RR[ii]))
		cc = aa - bb + 1.
		estim.append(cc)
	return estim

#-----------------------------------------------------------#
def jackknife_error(DSET1, DSET2, bins, key):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
	return 12

#-----------------------------------------------------------#
def estimator_calculator(d_count, r_count, dc, rc, drc, key):
#-----------------------------------------------------------#
# Calculates correlation estimators based on pair counts
# and number of data points given.
# Inputs:
#	d_count:
#	r_count:
#	dc:
#	rc:
#	drc:
#	key: how many estimators to calculate, integer
#		 1 = D&P simple
#		 2 = D&P simple and D&P
#		 3 = D&P simple and D&P and L&Z
#-----------------------------------------------------------#
	if (key == 1):	# do the simple davis and peebles estimator
		dp_simp = davis_peebles_simple(dc, rc, d_count, r_count)
		return dp_simp
	elif (key == 2):	# do the normal davis and peebles estimator
		dp_simp = davis_peebles_simple(dc, rc, d_count, r_count)
		dp_norm = davis_peebles(dc, drc, d_count, r_count)
		return (dp_simp, dp_norm)
	elif (key == 3):	# calculate all three estimators
		dp_simp = davis_peebles_simple(dc, rc, d_count, r_count)
		dp_norm = davis_peebles(dc, drc, d_count, r_count)
		lz		= landay_szalay(dc, rc, drc, d_count, r_count)
		return (dp_simp, dp_norm, lz)
	else:
		print("Invalid Key Input. TERMINATING.")
		exit()

#-----------------------------------------------------------#
def data_writer(bins, data, mults, file_name):
#-----------------------------------------------------------#
# Write a pandas dataframe to a csv.
# Includes bins, multipliers and data values
# data, mults, and bins must be numpy arrays
# file_name is the name of the file to which the data is
# to be written.
#-----------------------------------------------------------#
	df  = pd.DataFrame(data=data, index=mults, columns=bins)
	df.to_csv(file_name)

#-----------------------------------------------------------#
def data_reader(file_name):
#-----------------------------------------------------------#
# Reads a pandas dataframe from a csv.
#-----------------------------------------------------------#
	d_read   = pd.read_csv(file_name, header=None)
	dat_out  = retrieve_data(d_read)
	bins_out = retrieve_bins(d_read)
	mult_out = retrieve_multipliers(d_read)
	return (mult_out, bins_out, dat_out)

#-----------------------------------------------------------#
def retrieve_bins(dframe):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
	data = dframe.values
	bins = data[0,1::]
	return bins

#-----------------------------------------------------------#
def retrieve_multipliers(dframe):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
	data  = dframe.values
	mults = data[1::,0]
	return mults

#-----------------------------------------------------------#
def retrieve_data(dframe):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
	data = dframe.values
	estimator = data[1::,1::]
	return estimator

#-----------------------------------------------------------#
def big_average(array_in):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
	a1 = np.array(array_in)
	avg_array = [np.mean(a1[:,ii]) for ii in range(0, len(array_in[0]))]
	return avg_array


#-----------------------------------------------------------#
def covariance_matrix(estim_store, subd, bins):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
	avg   = np.array(big_average(estim_store))
	data  = np.array(estim_store)
	pref  = (subd-1)/subd
	outer = []
	dg    = []
	for ii in range(0, subd):
		#print("i = " + str(ii))
		i_mean = avg[ii]
		inner  = []
		for jj in range(0, bins):
			#print("j = " + str(jj))
			j_mean = avg[jj]
			tally  = 0
			for kk in range(0, subd):
				f_i    = data[kk,ii] - i_mean
				f_j    = data[kk,jj] - j_mean
				f_m    = f_i*f_j
				tally += f_m
			f_tally = pref*tally
			if (ii == jj):
				dg.append(f_tally)
				inner.append(f_tally)
			else:
				inner.append(f_tally)
		outer.append(inner)
	cov_mat = np.array(outer)
	diags   = np.array(dg)
	return [cov_mat, diags]

#############################################################
