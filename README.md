Dissertation Title: Clustering of MaNGA Galaxies

Author: Kian Alexander Shakerin

Supervisors: Dr. Rita Tojeiro and Mr. Chris Duckworth

Organizaiton: University of St Andrews

NOTE: The project is still underway and will be updated as it progresses. 

This is a repository for all the code I am writing for my dissertation.
The bulk of the project involves calculating correlation functions via pair counting.
Error estimates are also calculated using the jackknife error estimation method.

main.py - main program inside which all tests are run

modules.py - contains all the modules

plotting.py - contains some plotting modules to make data visualization easier

estimator_stability.py - tests the stability of the various correlation funciton estimators

jackknife.py - tests the jackknife error estimating method

cut_tests.py - splits galaxy populations into volume limited samples for CCF and ACF calculations

optimization_tests.py - modified k-d tree nearest neighbor search developed and tested (note: not stable)

property_ccfs.py - calculates CCFs for MaNGA data split on various resolved galactic properties: kinematics, etc.

volume_limited.py - more tests on volume limited galaxy samples
