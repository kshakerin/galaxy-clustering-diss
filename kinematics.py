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
DR7_data_filename         = "post_catalog.dr72bright0.fits"
DR7_randoms_filename0     = "random-0.dr72bright.fits"
DR7_randoms_filename1     = "random-1.dr72bright.fits"
DR7_randoms_filename10    = "random-10.dr72bright.fits"
MaNGA_targets_filename    = "MaNGA_targets_extNSA_tiled_ancillary.fits"
MaNGA_kinematics_filename = "mpl6_basic_kinematic_info.csv" #NOTE: CSV
MaNGA_properties_filename = "manga_firefly-v2_4_3-GLOBALPROP.fits"

# Imports the DR7 RA/DEC/z/mag
(RA_d, DEC_d, z_d, mr) = md.DR7_data_importer(data_directory + DR7_data_filename)
# Read in the Random data
filenames = [data_directory + DR7_randoms_filename0,
             data_directory + DR7_randoms_filename1,
             data_directory + DR7_randoms_filename10]
(RA_r, DEC_r) = md.DR7_randoms_importer(filenames)
# Read in the MaNGA Target Catalogue data
(RA_mng, DEC_mng, z_mng, mr_mng, IDs) = md.MANGA_data_importer(data_directory + MaNGA_targets_filename, 30)
#print(corr[0:5])
# Set the RA/DEC window for galaxy observations
(RA_low, RA_high, DEC_low, DEC_high) = md.initialize_window(180, 10, 10, 10)
win                                  = [RA_low, RA_high, DEC_low, DEC_high]
# Set bins
bin_num    = 15
bin_select = np.logspace(np.log10(0.1), np.log10(50), bin_num, endpoint=True)
x_axis     = (bin_select[1:] + bin_select[:-1])/2

print("total number of DR7 galaxies")
print(len(RA_d))
print("total number of random galaxies")
print(len(RA_r))


a_acf = [10.307599634301598, 10.242518907661431, 8.56171117337236, 7.071090630000395, 6.126643326528511, 4.888319528165291, 3.750665401180642, 2.5750105966217984, 1.5700063823437, 0.9926002763860589, 0.6297367530247335, 0.31517629292528415, 0.0030653453885260973, -0.2829551775078202]
m_acf = [16.4395091310141, 16.864311892951214, 16.400552669813024, 13.414791608546027, 10.026432352021244, 6.927159575734562, 3.7918973064361277, 2.1840389072742035, 1.3395748363208932, 0.892785594099589, 0.6480346627625833, 0.2519856700876275, -0.02272181830205222, -0.26860145419413073]

plt.figure(figsize=(15,15))
plt.plot(x_axis, a_acf, linewidth=2, markersize=12, label='Aligned (delta-PA < 30)')
plt.plot(x_axis, m_acf, linewidth=2, markersize=12, label='Misaligned (delta-PA > 30)')
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title('Kinematically Aligned vs Misalgned MaNGA Galaxy ACFs')
plt.xlabel("R [Mpc]")
plt.ylabel("Correlation Amplitude")
plt.legend()
plt.show()
exit()

# This Section Handles the Kinematics File
# Import data from the kinematics file
kinematics = []
with open(data_directory + MaNGA_kinematics_filename) as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        kinematics.append(row)
kinematics = np.array(kinematics[1::])
#kinematics = kinematics[:,1::]


kin = []
for rows in kinematics:
    hold = []
    hold.append(rows[0])
    for value in rows[1::]:
        hold.append(float(value))
    kin.append(hold)
print(kin[0])


# This Section Handles the MaNGA firefly File
# Import data from the  file
hdu1             = fits.open(data_directory + MaNGA_properties_filename)
manga_properties = hdu1[1].data
RA                = manga_properties.field(4)
DEC                = manga_properties.field(5)
z                = manga_properties.field(6)
plate_IFU        = manga_properties.field(1)

manga_properties = [[plate_IFU[ii], RA[ii], DEC[ii], z[ii]] for ii in range(0, len(plate_IFU))]
manga_properties = np.array(manga_properties)
kin              = np.array(kinematics)

combined_kinematic_data = []
combined_kinematic_data2 = []
for ii in range(0, len(kin)):
    prop = manga_properties[np.where(kin[ii,0] == manga_properties[:,0])]
    if len(prop) != 0:
        new_data = [float(prop[0][1]),
                    float(prop[0][2]),
                    float(prop[0][3]),
                    float(kin[ii,6]),
                    kin[ii,0]]
        new_data2 = [float(prop[0][1]),
                     float(prop[0][2]),
                     float(prop[0][3]),
                     float(kin[ii,6])]
        combined_kinematic_data.append(new_data)
        combined_kinematic_data2.append(new_data2)

print(combined_kinematic_data[0])
print("Length of combined data =  ", len(combined_kinematic_data))
print("Length of kinematic data = ", len(kin))
print("Length of MaNGA Galaxies = ", len(manga_properties))
# Match the kinematic properties to the galaxy poisitions using
# the plate-IFU ID tag

combined_kinematic_data2 = np.array(combined_kinematic_data2)
# Split the data into kinematically aligned and kinematically  misaligned galaxies
# Kinetmatic alignment is determined on a delta PA > 30 degrees split
# This is taken from Duckworth et al. 2018
delta_PA   = 30
aligned    = []
misaligned = []
for galaxy in combined_kinematic_data2:
    if galaxy[3] > 30:
        misaligned.append(galaxy)
    else:
        aligned.append(galaxy)

aligned    = np.array(aligned)
misaligned = np.array(misaligned)
print("Assinging random catalogue redshifts...")

random_galaxies     = [[RA_r[ii], DEC_r[ii]] for ii in range(0, len(RA_r))]
downsampled_randoms = [random.choice(random_galaxies) for ii in range(0, 10*len(combined_kinematic_data))]
z_random            = [random.choice(combined_kinematic_data2[:,2]) for ii in range(0, len(downsampled_randoms))]
final_randoms = []
for ii in range(0, len(downsampled_randoms)):
    RA, DEC = downsampled_randoms[ii]
    new = [RA, DEC, z_random[ii]]
    final_randoms.append(new)
final_randoms = np.array(final_randoms)


n_aligned     = len(aligned)
n_misaligned = len(misaligned)
n_random     = len(final_randoms)
print("Number of aligned galaxies:    " + str(n_aligned))
print("Number of misaligned galaxies: " + str(n_misaligned))
print("Number of random galaxies:     " + str(n_random))

print("Calculating aligned ACF...")
start = time.time()
aligned_ACF    = md.ACF_estimator(aligned[:,0:3], final_randoms, bin_select, 1)
print("Calculating misaligned ACF...")
misaligned_ACF = md.ACF_estimator(misaligned[:,0:3], final_randoms, bin_select, 1)
end = time.time() - start
print("Time taken to calculate ACFs: " + str(end))
print("Aligned ACF")
print(aligned_ACF)
print("Misaligned ACF")
print(misaligned_ACF)


plt.figure(figsize=(15,15))
plt.plot(x_axis, aligned_ACF, linewidth=2, markersize=12, label='Aligned')
plt.plot(x_axis, misaligned_ACF, linewidth=2, markersize=12, label='Misaligned')
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.title('Kinematically Aligned vs Misalgned MaNGA Galaxy ACFs')
plt.xlabel("R [Mpc]")
plt.ylabel("Correlation Amplitude")


plt.figure(figsize=(15,7))
plt.scatter(RA_mng, DEC_mng, color='b', s=1)
plt.scatter(RA_mng, DEC_mng, color='b', s=1)
plt.scatter(RA, DEC, color='r', s=1)
plt.show()

exit()

for name in file_names:
    hdu     = fits.open(name)
    rands   = hdu[1].data
    RA      = rands.field(0)
    DEC     = rands.field(1)
    RA_out  = np.append(RA_out, RA)
    DEC_out = np.append(DEC_out, DEC)



# END #
