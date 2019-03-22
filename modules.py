import numpy as np
import math as mt
import pandas as pd
import os
from astropy.io import fits
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy import units as uni
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
def setup_data_and_randoms(data, randoms, win):
#-----------------------------------------------------------#
# Takes in a set of galaxies and a set of randoms with
# with a given RA/DEC window and returns all the data
# and randoms inside that window with the randoms
# assigned redshifts to match the data population redshifts.
# Return the positions in both spherical and cartesian forms.
# Inputs:
# win: window parameters - [RA_low, RA_high, DEC_low, DEC_high]
# Outputs:
#-----------------------------------------------------------#
	# Set up data RA/DEC/z
	RA_dat  = data[:,0]
	DEC_dat = data[:,1]
	z_dat   = data[:,2]
	# Set up randoms RA/DEC
	RA_ran  = randoms[:,0]
	DEC_ran = randoms[:,1]
	# Get data from selected window
	(RA1, DEC1, z1) = data_selector(RA_dat, DEC_dat, z_dat,
				 			        win[0], win[1], win[2], win[3])
	data_spherical  = np.array([np.array([RA1[ii], DEC1[ii], z1[ii]])
							   for ii in range(0, len(RA1))])
	data_cartesian  = np.array([spherical_to_cartesian(ii[0], ii[1], ii[2])
							   for ii in data_spherical])
	# Set up the ransoms from the selected window
	(RA2, DEC2, z2) = random_selector(RA_ran, DEC_ran, z1,
								      win[0], win[1], win[2], win[3])
	rand_spherical  = np.array([np.array([RA2[ii], DEC2[ii], z2[ii]])
							   for ii in range(0, len(RA2))])
	rand_cartesian  = np.array([spherical_to_cartesian(ii[0], ii[1], ii[2])
							   for ii in rand_spherical])
	# Return the 4 2D arrays of data
	return data_spherical, data_cartesian, rand_spherical, rand_cartesian


#-----------------------------------------------------------#
def DR7_data_importer(file_name):
#-----------------------------------------------------------#
# Imports RA/DEC/Z/MAG from the DR7 data file
#-----------------------------------------------------------#
	hdu1     = fits.open(file_name)
	DR7_data = hdu1[1].data
	RA       = DR7_data.field(1)
	DEC      = DR7_data.field(2)
	Z        = DR7_data.field(9)
	abs_m_d  = DR7_data.field(8)
	MAG      = np.array([abs_m_d[i][2] for i in range(0,len(abs_m_d))])
	return RA, DEC, Z, MAG

#-----------------------------------------------------------#
def DR7_randoms_importer(file_names):
#-----------------------------------------------------------#
# Imports the RA and DEC for a list of DR7 random galaxy files
#-----------------------------------------------------------#
	RA_out  = []
	DEC_out = []
	for name in file_names:
		hdu     = fits.open(name)
		rands   = hdu[1].data
		RA      = rands.field(0)
		DEC     = rands.field(1)
		RA_out  = np.append(RA_out, RA)
		DEC_out = np.append(DEC_out, DEC)
	return RA_out, DEC_out

#-----------------------------------------------------------#
def MANGA_data_importer(file_name, extra_data):
#-----------------------------------------------------------#
# Imports RA/DEC/Z/MAG from the MaNGA data file
#-----------------------------------------------------------#
	hdu               = fits.open(file_name)
	manga_target_data = hdu[1].data
	RA                = manga_target_data.field(0)
	DEC               = manga_target_data.field(1)
	z                 = manga_target_data.field(2)
	abs_m_mng         = manga_target_data.field(5)
	extra    	 	  = manga_target_data.field(extra_data)
	#print(abs_m_mng)
	mr                = np.array([abs_m_mng[i][2] for i in range(0,len(abs_m_mng))])
	return RA, DEC, z, mr, extra

#-----------------------------------------------------------#
def initialize_window(ra_start, dec_start, delt_ra, delt_dec):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
	RA_max  = ra_start + delt_ra
	DEC_max = dec_start + delt_dec
	return ra_start, RA_max, dec_start, DEC_max

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
def sphere_to_cart_list(dset):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
	output = []
	for pos in dset:
		theta = pos[0]
		phi   = pos[1]
		zval  = pos[2]
		output.append(spherical_to_cartesian(theta, phi, zval))
	return np.array(output)

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
def ACF_estimator(dset1, rset1, bins, key):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
	N_data = len(dset1)
	N_rand = len(rset1)
	if (key==0):
		DD  = auto_sort_pair_counter(dset1, bins, 0)
		RR  = auto_sort_pair_counter(rset1, bins, 0)
		ACF = davis_peebles_simple(DD, RR, N_data, N_rand)
		return ACF
	elif (key==1):
		DD  = auto_sort_pair_counter(dset1, bins, 0)
		DR  = improved_cross_correlate(dset1, rset1, bins, 0)
		ACF = davis_peebles(DD, DR, N_data, N_rand)
		return ACF
	elif (key==2):
		DD  = auto_sort_pair_counter(dset1, bins, 0)
		RR  = auto_sort_pair_counter(rset1, bins, 0)
		DR  = improved_cross_correlate(dset1, rset1, bins, 0)
		ACF = landay_szalay(DD, RR, DR, N_data, N_rand)
		return ACF
	else:
		return [0 for i in bins[1:-1]]

#-----------------------------------------------------------#
def CCF_estimator(dset1, dset2, rset2, bins):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
	N_data = len(dset2)
	N_rand = len(rset2)
	D1D2   = improved_cross_correlate(dset1, dset2, bins, 0)
	D1R2   = improved_cross_correlate(dset1, rset2, bins, 0)
	CCF    = davis_peebles(D1D2, D1R2, N_data, N_rand)
	return CCF

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
	for ii in range(0, bins):
		i_mean = avg[ii]
		inner  = []
		for jj in range(0, bins):
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

#-----------------------------------------------------------#
def data_selector_prop(RA, DEC, Z, prop, RA_l, RA_h, DEC_l, DEC_h):
#-----------------------------------------------------------#
# Selects some data in a given RA/DEC range and outputs the
# RA/DEC pairs with the corresponding redshifts and property.
# Return in the form of a list 4xN
#-----------------------------------------------------------#
	# Slice data on RA
	ind1   = np.where((RA > RA_l) & (RA < RA_h))
	RA_v1  = RA[ind1]
	DEC_v1 = DEC[ind1]
	Z_v1   = Z[ind1]
	p_v1   = prop[ind1]
	# Slice data on DEC
	ind2   = np.where((DEC_v1 > DEC_l) & (DEC_v1 < DEC_h))
	RA_vf  = RA_v1[ind2]
	DEC_vf = DEC_v1[ind2]
	Z_vf   = Z_v1[ind2]
	p_zf   = p_v1[ind2]
	# Output in the format [RA_i, DEC_i, Z_i, prop_i]
	output = [[RA_vf[ii], DEC_vf[ii], Z_vf[ii], p_zf[ii]] for ii in range(0, len(RA_vf))]
	return output

#-----------------------------------------------------------#
def remove_from_catalogues(input_cat, main_cat):
#-----------------------------------------------------------#
# Removes data points from overlapping catalogues in order
# to computer cross correlation functions. Takes in arrays
# with rrows: [RA, DEC, z].
# Inputs:
#	input_cat: input catalogue
#	main_cat: main catalogue
# Output:
#	new_data: copy of main catalogue with all the input
#			  catalogue galaxies removed
#	matches: number of overlapping galaies between the two
#			 catalogues
#-----------------------------------------------------------#
	indexes = []
	for jj in input_cat:
		aaa = np.where(main_cat[:,:] == jj)[0]
		if len(aaa) == 1:
			indexes.append(aaa.tolist())
		elif len(aaa) != 0:
			c1 = spherical_to_cartesian(jj[0], jj[1], jj[2])
			abs_dist = []
			for ii in aaa:
				c2 = spherical_to_cartesian(main_cat[ii,0],
											main_cat[ii,1],
											main_cat[ii,2])
				dist = distance_cartesian(c1, c2)
				abs_dist.append(dist)
			indexes.append(aaa[np.where(abs_dist==min(abs_dist))].tolist())
	matches = len(indexes)
	new_data = np.delete(main_cat, indexes, axis=0)
	return new_data, matches

#-----------------------------------------------------------#
def coordinate_match(dset1, dset2):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
# Matches galaxies from dset1 to galaxies in dset2.
# Returns a copy of dset2 with all the matched
# galaxies removed.
	# set up the coordinate systems
	coo_dset1 = SkyCoord(dset1[:,0]*uni.deg, dset1[:,1]*uni.deg)
	coo_dset2 = SkyCoord(dset2[:,0]*uni.deg, dset2[:,1]*uni.deg)
	# match the galaxies
	idx_dset2, d2d_dset2, d3d_dset2 = coo_dset1.match_to_catalog_sky(coo_dset2)
	# remove only matches within 3 arcsec
	new_dset2 = np.delete(dset2, idx_dset2[d2d_dset2.arcsec < 3], axis=0)
	return new_dset2

#-----------------------------------------------------------#
def volume_limited(data, limit):
#-----------------------------------------------------------#
# Luminosity Thresholding
#-----------------------------------------------------------#
	z_lowest = -50	# hardcode a lower limit to account for errors in data
	# find the max redshift
	indexes1       = np.where(data[:,3] > limit)
	above_cut      = data[indexes1]
	zmin           = max(above_cut[:,2])
	# find the subsamples
	indexes2       = np.where(data[:,3] < limit)
	mag_lim_sample = data[indexes2]
	vol_subsample  = mag_lim_sample[np.where((mag_lim_sample[:,2] < zmin) &
											 (mag_lim_sample[:,2] > z_lowest))]
	# return volume limited sample
	return vol_subsample, zmin, limit

#-----------------------------------------------------------#
def volume_limited_bins(data, limits):
#-----------------------------------------------------------#
# Luminosity Bins
#-----------------------------------------------------------#
	z_lowest = -50	# hardcode a lower limit to account for errors in data
	up		 = limits[0]
	down	 = limits[1]
	# find the max redshift
	indexes1       = np.where(data[:,3] > up)
	above_cut      = data[indexes1]
	zmax           = max(above_cut[:,2])
	# find the minimum redshift
	indexes2       = np.where(data[:,3] < down)
	below_cut      = data[indexes2]
	zmin           = min(below_cut[:,2])
	# find the subsamples
	indexes3       = np.where((data[:,3] < up) & (data[:,3] > down))
	mag_lim_sample = data[indexes3]
	vol_subsample  = mag_lim_sample[np.where((mag_lim_sample[:,2] > zmin) &
											 (mag_lim_sample[:,2] < zmax))]
	# return volume limited sample
	return vol_subsample, zmin, zmax, limits

#-----------------------------------------------------------#
def ACF_jackknife(dset1, rset1, bins, win, subs, key):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
# Data sets are input using cartesian coordinates
	# First we calculate the correlation function on the whole data set
	data_cart = sphere_to_cart_list(dset1)
	rand_cart = sphere_to_cart_list(rset1)
	ACF_main = ACF_estimator(data_cart, rand_cart, bins, key)
	print(ACF_main)
	# Next we split a set of data and randoms into N subvolumes
	RA_subs  = np.linspace(win[0], win[1], subs[0]+1, endpoint=True)
	DEC_subs = np.linspace(win[2], win[3], subs[1]+1, endpoint=True)
	data_subs = []	# arrays for storing the subdivisions
	rand_subs = []
	for ii in range(subs[0]):
		RA_l = RA_subs[ii]		# Initialize the subdivision window
		RA_h = RA_subs[ii+1]	# boundaries.
		for jj in range(subs[1]):
			DEC_l = DEC_subs[jj]	# Initialize the subdivision window
			DEC_h = DEC_subs[jj+1]	# boundaries.
			# Select the data corresponding to a subsection
			d_i   = dset1[np.where((dset1[:,0] > RA_l) & (dset1[:,0] < RA_h))]
			d_f   = d_i[np.where((d_i[:,1] > DEC_l) & (d_i[:,1] < DEC_h))]
			d_crt = sphere_to_cart_list(d_f)	# convert to cartersian
			print(len(d_crt))
			data_subs.append(d_crt)
		    # Select the randoms corresponding to a subsection
			r_i   = rset1[np.where((rset1[:,0] > RA_l) & (rset1[:,0] < RA_h))]
			r_f   = r_i[np.where((r_i[:,1] > DEC_l) & (r_i[:,1] < DEC_h))]
			r_crt = sphere_to_cart_list(r_f)	# convert to cartesian
			rand_subs.append(r_crt)
	# Now we perform the jackknife error estimate
	ACF_store = []
	for ii in range(0, len(data_subs)):
		print("N: " + str(ii))
		data_hold = []
		rand_hold = []
		start = time.time()
		for jj in range(0, len(data_subs)):
			if (jj == ii):
				continue
			data_hold = data_hold + data_subs[jj].tolist()
			rand_hold = rand_hold + rand_subs[jj].tolist()
		ACF_i = ACF_estimator(np.array(data_hold), np.array(rand_hold), bins, 1)
		print("Time taken for 1 calculation:", time.time() - start)
		ACF_store.append(ACF_i)
	# Calculate the covariance matrix
	(cv_matrix, diags) = covariance_matrix(ACF_store, subs[0]*subs[1], len(bins)-1)
	return ACF_main, ACF_store, cv_matrix, diags

#-----------------------------------------------------------#
def volume_limited_ACF(dset1, rset1, bins, win, mag_limits, key):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
	dat_sp_a = []
	dat_c_a  = []
	ran_sp_a = []
	ran_c_a  = []
	v_subs   = []
	estims   = []
	for mag in mag_limits:
		print(mag)
		(vsub, _, _) = volume_limited(dset1, mag)
		v_subs.append(vsub)
		(data_sp, data_c, rand_sp, rand_c) = setup_data_and_randoms(vsub, rset1, win)
		dat_sp_a.append(data_sp)
		dat_c_a.append(data_c)
		ran_sp_a.append(rand_sp)
		ran_c_a.append(rand_c)
		estims.append(ACF_estimator(data_c, rand_c, bins, 1))
	return 	estims, dat_sp_a, dat_c_a, ran_sp_a, ran_c_a, v_subs

#-----------------------------------------------------------#
def volume_limited_CCF(dset1, dset2, rset2, bins, win, mag_limits):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
	# Begin by removing all overlapping galaxies
	new_dset2  = coordinate_match(dset1, dset2)
	# Initialize all the output arrays
	CCF_est    = []
	vsub1_a    = []
	vsub2_a    = []
	dset1_sp_a = []
	dset2_sp_a = []
	rset2_sp_a = []
	# Do the CCF estimates for every magnitude limited sample
	for mag in mag_limits:
		print(mag)
		# Split all the data appropriately and perform calculationns
		(vsub_d1, _, _) 					   = volume_limited(dset1, mag)
		(vsub_d2, _, _) 					   = volume_limited(new_dset2, mag)
		(dset1_sp, dset1_c, _, _) 			   = setup_data_and_randoms(vsub_d1, rset2, win)
		(dset2_sp, dset2_c, rset2_sp, rset2_c) = setup_data_and_randoms(vsub_d2, rset2, win)
		est 								   = CCF_estimator(dset1_c, dset2_c, rset2_c, bins)
		CCF_est.append(est)
		vsub1_a.append(vsub_d1)
		vsub2_a.append(vsub_d2)
		dset1_sp_a.append(dset1_sp)
		dset2_sp_a.append(dset2_sp)
		rset2_sp_a.append(rset2_sp)
	return CCF_est, dset1_sp_a, dset2_sp_a, rset2_sp_a, vsub1_a, vsub2_a

#############################################################
