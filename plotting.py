import matplotlib.pyplot as plt
import modules as md
import numpy as np
import math as mt
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
#                          PLOTS                            #
#-----------------------------------------------------------#
#-----------------------------------------------------------#
#############################################################

#-----------------------------------------------------------#
def sky_positions_1(RA_d, DEC_d):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
    # Plot sky positions
    plt.figure(figsize=(15,7))
    plt.scatter(RA_d, DEC_d, marker='o', s=1)
    plt.title("Galaxy Catelogue Sky Positions")
    plt.xlabel("RA [deg]")
    plt.ylabel("DEC [deg]")

#-----------------------------------------------------------#
def z_histogram_1(z_d):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
    # Plot histogram
    plt.figure(figsize=(10,6))
    plt.hist(z_d, bins=30)
    plt.title("Histogram of Redshifts")
    plt.xlabel("Redshift")
    plt.ylabel("Number of Galaxies")

#-----------------------------------------------------------#
def sky_positions_2(RA_array, DEC_array):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
    # Plot the sky positions of the random data and the normal data
    plt.figure(figsize=(15,7))
    plt.scatter(RA_array[1], DEC_array[1], marker="o", s=1, label="randoms")
    plt.scatter(RA_array[0], DEC_array[0], marker="o", s=1, label="data")
    plt.legend()
    plt.title("Galaxy and Random Catalogue Sky Poisitions")
    plt.xlabel("RA [deg]")
    plt.ylabel("DEC [deg]")

#-----------------------------------------------------------#
def z_histogram_2(z_r):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
    # Plot a histogram of the new redshifts
    plt.figure(figsize=(10,6))
    plt.hist(z_r, bins=30)
    plt.title("Histogram of Random Redshifts")
    plt.xlabel("Redshift")
    plt.ylabel("Number of Galaxies")

#-----------------------------------------------------------#
def z_histogram_3(z_d, z_r, den):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
    # Plot a histgram of both the data redshifts and the random redshifts
    plt.figure(figsize=(10,6))
    plt.hist(z_d, bins=300, density=den, label="Data")
    plt.hist(z_r, bins=300, density=den, label="Random", histtype="step", lw=1)
    plt.legend()
    plt.title("Histograms of Data and Random Redshifts")
    plt.xlabel("Redshift")
    plt.ylabel("Number of Galaxies")


#-----------------------------------------------------------#
def sky_positions_3(RA_array, DEC_array):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
    # Plot of the DR7 data with selctions from data and random RA/DEC
    # overlaid
    RA_d  = RA_array[0]
    RA_1  = RA_array[1]
    RA_2  = RA_array[2]
    DEC_d = DEC_array[0]
    DEC_1 = DEC_array[1]
    DEC_2 = DEC_array[2]
    plt.figure(figsize=(15,7))
    plt.scatter(RA_d, DEC_d, marker="o", s=1, label="data")
    plt.scatter(RA_1, DEC_1, marker="o", s=1, label="data selection")
    plt.scatter(RA_2, DEC_2, marker="x", s=1, label="random selection")
    plt.title("Spatial Distributions of Selections")
    plt.xlabel("RA [deg]")
    plt.ylabel("DEC [deg]")
    plt.legend()

#-----------------------------------------------------------#
def z_histogram_4(z_d, z_1, z_2):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
    # Histogram of redshifts for the data and random selections compared
    # to the main data set
    plt.figure(figsize=(10,6))
    plt.hist(z_d, bins=30, density=True, label="main data")
    plt.hist(z_1, bins=30, density=True, label="data subset")
    plt.hist(z_2, bins=30, density=True, label="random subset", histtype="step", lw=1)
    plt.legend()
    plt.title("Density Normalized Redshift Histogram")
    plt.xlabel("Redshift")
    plt.ylabel("Number of Galaxies")


#-----------------------------------------------------------#
def estimator_plots(bins, values, r_mults, est_name):
#-----------------------------------------------------------#
#-----------------------------------------------------------#
    plt.figure(figsize=(15,7))
    for ii in range(0, len(r_mults)):
    	plt.plot(bins, values[ii], linewidth=2, markersize=12,
    			label=str(r_mults[ii]))
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.title(est_name)
    plt.xlabel("R [Mpc]")
    plt.ylabel("Correlation Amplitude")
    plt.legend()


#############################################################
#END
