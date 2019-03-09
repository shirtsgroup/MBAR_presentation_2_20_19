# This Python script contains example NaCl simulations in OpenMM, 
# followed by analysis with MBAR.

# More specifically, this script contains a protocol to evaluate the free energies,
# expectation value of the Na-Cl distance, and uncertainty in each of these measurements
# at a temperature of 400 K, while varying: 
#
# 1) the number of states (# of temperatures for which we are performing NVT dynamics (Langevin))
#
# 2) the total number of samples supplied as input when performing MBAR analysis.

# The protocol is broken up into the following sections:

# 1) Run-time options
# 2) Define objects that will store our data
# 3) Prepare and run OpenMM simulations (Langevin dynamics @ named temps.)
# 4) Sub-sample simulation data to obtain decorrelated samples
# 5) Calculate 'weights' for each thermodynamic state (temp.) with MBAR
# 6) Calculate dimensionless free energies with MBAR
# 7) Calculate < R_Na-Cl > with MBAR
# 8) Analyze MBAR uncertainty in < R_Na-Cl > as a function of ensemble size
# 9) Plot results for T = 400 K

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from os import getcwd as cwd
import numpy as np
import mdtraj as md
import matplotlib.pyplot as pyplot
import math
import util
import random
import multiprocessing
import subprocess
from multiprocessing import Pool
from openmmtools.testsystems import SodiumChlorideCrystal
from pymbar import MBAR, timeseries
from zipfile import ZipFile
from shutil import rmtree

print("Running the MBAR free energy protocol for sodium chloride...")

#############
#
# 1) Run-time options
#
#############

###
#
# System settings
#
###

# Default number of processors is the total available on the system - 2
processors = 10
# Define output directories for different run settings/locations
if cwd().split('/')[-1] == 'FreeEnergy':
 output_dir = 'output/'
 figures_dir = 'figures/'
if cwd().split('/')[-1] == 'MBAR_presentation_2_20_19':
 output_dir = 'NaCl/FreeEnergy/output/'
 figures_dir = 'NaCl/FreeEnergy/figures/'

###
#
# Script-specific settings
#
###

# For each number of states, 'num_states', we discretize the temperature range from 300 to 500 K into that number of states
# For example, if num_states=2, we run 2 Langevin dynamics simulations: one simulation at 300 K and one at 500 K
# if num_states=3 we run simulations at 300 K, 400 K, and 500 K
# etc...

state_range = [num_states for num_states in range(2,50)] # 'state_range' contains a range of total states for which we will run Langevin dynamics simulations
min_temp = 300.0 # Units = Kelvin
max_temp = 500.0 # Units = Kelvin
zip_file_name = "data.zip"
search_for_existing_data = True # If set to 'False', we won't look for old simulation data before running a new simulation, when 'simulation_data_exists' = True
simulation_data_exists = False # If set to 'False', we will run Langevin dynamics simulations with OpenMM for all temperatures of interest
compress_simulation_data = False # Compress decorrelated simulation data into a zip file
analysis_data_exists = False # If set to 'False', we will analyze the simulation data with MBAR

# In order to vary the total number of samples in our ensemble, 
# while including a uniform number of samples from each state, 
# we follow this procedure:
#
# 1) Run simulations for all temperatures in the current 'state_range'
# 2) Decorrelate the timeseries data with MBAR
# 3) Identify the state (timeseries/temperature) that has the smallest number of decorrelated samples
# 4) Calculate the total number of samples that will be drawn from each state as:
#
#    sample_fraction * min(len(all_decorrelated_timeseries))

sample_fraction_step = 0.05 # Defines the step size by which we incremently increase 'sample_fraction'
# 'sample_fraction_range' contains the range of 'sample_fraction's for which we sub-sample all timeseries,
# where the range is composed of integer indices, which are later multiplied by 'sample_fraction_step' to
# calculate 'sample_fraction'
sample_fraction_range = range(0,7)
min_fraction = 0.2 # Defines the minimum fraction by which to sub-sample our time-series in order to evaluate the relationship between sample size and uncertainty in MBAR
sub_sampling_levels = [min_fraction + i * sample_fraction_step for i in sample_fraction_range]
###
#
# OpenMM simulation settings
#
###

simulation_time_step = 0.002 # Units = picoseconds
kB = 0.008314462  #Boltzmann constant (Gas constant) in kJ/(mol*K)
simulation_steps = 1000000 # Number of steps used in individual Langevin dynamics simulations
print_frequency = 10 # Number of steps to skip when printing output
total_simulation_time = simulation_time_step * simulation_steps # Units = picoseconds

###
#
# MBAR analysis settings
#
###

nskip = 10 # Number of steps to skip when reading trajectory to decorrelate a timeseries with MBAR

#############
#
# 2) Define objects that will store our data
#
#############

free_energies_for_each_num_states = [] # stores free energy predictions from MBAR for each total # of states
free_energies_for_each_num_states_same_total_samples = [] # stores free energy predictions from MBAR for each total # of states, with the same total # of samples
uncertainties_for_each_num_states = [] # stores uncertainties in free energy predictions from MBAR for each total # of states
uncertainties_for_each_num_states_same_total_samples = [] # stores uncertainties in free energy predictions from MBAR for each total # of states, with the same total # of samples
distances_for_each_num_states = [] # stores decorrelated R_Na-Cl simulation data for each total # of states
distances_for_each_num_states_same_total_samples = [] # stores decorrelated R_Na-Cl simulation data for each total # of states, with the same total # of samples
distance_averages_for_each_num_states = [] # stores distance predictions from MBAR for each total # of states
distance_averages_for_each_num_states_same_total_samples = [] # stores distance predictions from MBAR for each total # of states, with the same total # of samples
uncertainty_distances_for_each_num_states = [] # stores uncertainties in distance predictions from MBAR for each total # of states
uncertainty_distances_for_each_num_states_same_total_samples = [] # stores uncertainties in distance predictions from MBAR for each total # of states, with the same total # of samples
weights_for_each_num_states = [] # stores 'weights' (output from MBAR) for each total # of states
weights_for_each_num_states_same_total_samples = [] # stores 'weights' (output from MBAR) for each total # of states, with the same total # of samples
temperatures_for_each_num_states = [] # stores the temperatures for which we run Langevin dynamics simulations
average_uncertainty_for_each_fraction_of_samples = [[0.0 for num_states in state_range] for index in sub_sampling_levels] # Contains the average uncertainty in < R_Na-Cl > (for the range of total states used) for each fraction used to sub-sample the time-series
average_distance_for_each_fraction_of_samples = [[0.0 for num_states in state_range] for index in sub_sampling_levels] # Contains the average < R_Na-Cl > (for the range of total states used) for each fraction used to sub-sample the time-series

#############
#
# 3) Prepare and run OpenMM simulation
#
#############

def individual_simulation_procedure(temperature):

###
#
# This subroutine performs an OpenMM Langevin dynamics simulation
# at the supplied 'temperature'
#
###

# OpenMM expects three objects in order to run a simulation: a 'system', an 'integrator', and a 'context'
   system = SodiumChlorideCrystal() # Define a system
   integrator = LangevinIntegrator(temperature, total_simulation_time, simulation_time_step) # Define Langevin integrator
   simulation = Simulation(system.topology, system.system, integrator) # Define a simulation 'context'
#   if os.path.exists(str(output_dir+str(temperature))) and not(search_for_existing_data): os.remove(str(output_dir+str(temperature)))
   if not(os.path.exists(str(output_dir+str(temperature)))): os.mkdir(str(output_dir+str(temperature))) # Create a directory for the output if it doesn't already exist
   simulation.reporters.append(PDBReporter(str(output_dir+str(temperature)+"/coordinates.pdb"),print_frequency)) # Write simulation PDB coordinates
   simulation.reporters.append(StateDataReporter(str(output_dir+str(temperature)+"/sim_data.dat"), print_frequency, \
   step=True, totalEnergy=True, potentialEnergy=True, kineticEnergy=True, temperature=True)) # Write simulation data
   simulation.context.setPositions(system.positions) # Assign particle positions for this context
   simulation.minimizeEnergy() # Set the simulation type to energy minimization
   simulation.step(simulation_steps) # Run the simulation
   return

#############
#
# 4) Sub-sample simulation data to obtain decorrelated samples
#
#############

def individual_analysis_procedure(temperature):

###
#
# This subroutine analyzes a timeseries for 'temperature',
# and generates a set of decorrelated sample energies and distances,
# which are used in later sampling to generate a free energy surface.
#
###
 if (search_for_existing_data and not(os.path.exists(str(output_dir+str(temperature)+"/uncorrelated_distances.dat")))) or not(search_for_existing_data):
   output_obj = open(str(output_dir+str(temperature)+"/sim_data.dat"),'r')
   E_total_all_temp = np.array([l.split(',')[3] for l in output_obj.readlines()]) # E_total_all_temp temporarily stores the total energies from NaCl simulation output
   output_obj.close()
   distances = util.get_distances(str(output_dir+str(temperature)+"/coordinates.pdb"),simulation_steps) # Read in the distances
   E_total_all = np.array(np.delete(E_total_all_temp,0,0),dtype=float) # E_total_all stores total energies from NaCl simulation output, after re-typing
   [t0, g, Neff_max] = timeseries.detectEquilibration(E_total_all,nskip=nskip) # Identify the indices of samples with high statistical efficiency (g)
   E_total_equil = E_total_all[t0:] # Using the index for the equilibration time (t0), truncate the time-series data before this index
   uncorrelated_energy_indices = timeseries.subsampleCorrelatedData(E_total_equil, g=g) # Determine indices of uncorrelated samples
   np.savetxt(str(output_dir+str(temperature)+'/uncorrelated_total_energies.dat'),E_total_equil[uncorrelated_energy_indices]) # Write uncorrelated total energies to file
   np.savetxt(str(output_dir+str(temperature)+'/uncorrelated_distances.dat'),distances[uncorrelated_energy_indices]) # Write uncorrelated Na-Cl distances to file
   return

def read_simulation_data(temperature):

###
#
# Read the decorrelated reduced potential energies and Na-Cl distances
# from OpenMM simulation output for 'temperature'.
#
###

   U_uncorrelated = np.array([]) # Uncorrelated total energies
   distances_obj = open(str(output_dir+str(temperature)+'/uncorrelated_distances.dat'),'r')
   distances = np.array([float(l) for l in distances_obj.readlines()]) # Read decorrelated distances from file
   distances_obj.close()
   uncorrelated_energies_obj = open(str(output_dir+str(temperature)+'/uncorrelated_total_energies.dat'),'r')
   U_uncorrelated = np.append(U_uncorrelated,[float(l) for l in uncorrelated_energies_obj.readlines()]) # Read decorrelated total energies from file
   uncorrelated_energies_obj.close()
   return(U_uncorrelated,distances)

def zip_files(source_folder,zip_file_dest):
# Zip files to reduce size
 with ZipFile(zip_file_dest,'w') as zip:
  for root, dirs, files in os.walk(str()):
   for file in files:
    zip.write(os.path.join(root,file))
 zip.close()

def unzip_files(zip_file,destination_folder):
# extract zip archive
 with ZipFile(zip_file,'r') as zip:
  zip.extractall(destination_folder)
 zip.close()

def remove_simulation_data():
   subprocess.call(['rm','-rf',str(output_dir),'*/coordinates.pdb'])
   subprocess.call(['rm','-rf',str(output_dir),'*/sim_data.dat'])
   return

def search_for_data(temperature_array,filename):
   for index in range(0,len(temperature_array)):
    if os.path.exists(str(output_dir+str("/")+str(temperature_array[index])+str("/")+str(filename))):
     temperature_array.remove(temperature_array[index])
   return(temperature_array)
   

if __name__ == '__main__':

#############
#
# Begin iteration over states
#
#############

# If we haven't run simulations yet we start here
 if not(simulation_data_exists):

# Begin iteration over states
  for num_states in state_range:
   print("Running simulations with "+str(num_states)+" states")
#  Define the temperature range based upon the number of states
   simulation_temperatures = [round((min_temp + i*(max_temp-min_temp)/(num_states-1)),1) for i in range(0,num_states)]
# Look for existing simulation data?
   if search_for_existing_data == True: 
     simulation_temperatures = search_for_data(simulation_temperatures)

# Run simulations in parallel
   pool = Pool(processes=processors)
   pool.map(individual_simulation_procedure,simulation_temperatures)
 
 if compress_simulation_data:
   zip_files(output_dir,str(os.getcwd()+str("/")+zip_file_name))

# If we haven't analyzed simulation data yet we start here:
 if not(analysis_data_exists):
# Begin iteration over states
  for num_states in state_range:
   temperatures = [round((min_temp + i*(max_temp-min_temp)/(num_states-1)),1) for i in range(0,num_states)]
# Run timeseries analysis in parallel
   if search_for_existing_data == True:
    simulation_temperatures = [round((min_temp + i*(max_temp-min_temp)/(num_states-1)),1) for i in range(0,num_states)]
    for temperature in simulation_temperatures:
     if os.path.exists(str(output_dir+str(temperature))):
      simulation_temperatures.remove(temperature)
    if len(simulation_temperatures) > 0:
     print("Running analysis with "+str(num_states)+" states")
     pool = Pool(processes=processors)
     pool.map(individual_analysis_procedure,simulation_temperatures)
   else:
     pool = Pool(processes=processors)
     pool.map(individual_analysis_procedure,temperatures)
 else:
  if compress_simulation_data: unzip_files(str(os.getcwd()+str("/")+zip_file_name),output_dir)

#############
#
# BEGIN ITERATION OVER FRACTION INDEX (ENSEMBLE SIZE)
#
#############

 fraction_index = 0
 for fraction_of_samples in sub_sampling_levels:

#############
#
# BEGIN ITERATION OVER EACH NUMBER OF STATES
#
#############

  num_states_index = 0
  for num_states in state_range:
   print("with "+str(num_states)+" states...")
# Create temporary storage arrays
   state_counts = [] # Stores the state distribution for an ensemble
   state_counts_same_total_samples = [] # Stores the state distribution when the total number of samples is held constant for each number of states
   U_uncorrelated = [] # Stores the decorrelated total energies read in for all states in the range(0,num_states)
   U_uncorrelated_same_total_samples = [] #Stores the decorrelated total energies read in for all states in the range(0,num_states), when the total # of samples is constant
   distances = [] # Stores the decorrelated distances read in for all states in the range(0,num_states)
   distances_same_total_samples = [] # Stores the decorrelated distances read in for all states in the range(0,num_states), when the total # of samples is constant
   # Define the temperatures that we are sampling based on 'num_states'
   temperatures = [round((min_temp + i*(max_temp-min_temp)/(num_states-1)),1) for i in range(0,num_states)]
   temperatures_for_each_num_states.extend([temperatures])
# When sub-sampling to the same total number of samples for each state distribution we determine which trajectory has the smallest number of decorrelated samples, and multiply this value by 'fraction_of_samples' to get total samples
# Stated differently, we use min('num_states') data to determine 'total_smaples'
   if num_states == 2:
    total_samples = int(round(2.0*fraction_of_samples*min(len(read_simulation_data(temperatures[0])[0]),len(read_simulation_data(temperatures[0])[0]))))
    print("Total samples ="+str(total_samples))
   samples_per_state = int(round(total_samples/num_states)-1) # defines the number of samples drawn from each temperature ensemble whe the total # of samples is constant
# Read the data for random, decorrelated samples, and obtain 'total_smaples' for each temperature/state
   for temperature in temperatures:
    U_uncorrelated_temp, distances_temp = read_simulation_data(temperature) 
    U_uncorrelated.extend(U_uncorrelated_temp)
    sample_indices = []
    while len(sample_indices) < samples_per_state:
     sample_index = random.randint(0,len(U_uncorrelated_temp)-1)
     if sample_index not in sample_indices: 
      sample_indices.extend([int(sample_index)])
      U_uncorrelated_same_total_samples.extend([U_uncorrelated_temp[sample_index]])
      distances_same_total_samples.extend([distances_temp[sample_index]])
    state_counts.extend([len(U_uncorrelated_temp)])
    state_counts_same_total_samples.extend([samples_per_state])
    distances.extend(distances_temp)

#############
#
# 5) Calculate 'weights' for each thermodynamic state (temp.) with MBAR
#
#############

# 'u_kn' contains the decorrelated reduced potential energies evaluated at each temperature (state)
   u_kn = np.array([[float(float(U_uncorrelated[sample])/float(temperature)) for sample in range(0,len(U_uncorrelated))] for temperature in temperatures])
   u_kn_same_total_samples = np.array([[float(float(U_uncorrelated_same_total_samples[sample])/float(temperature)) for sample in range(0,len(U_uncorrelated_same_total_samples))] for temperature in temperatures])
# Initialize MBAR
   mbar = MBAR(u_kn, state_counts, verbose=False, relative_tolerance=1e-12)
# Initialize MBAR using u_kn_same_total_samples, constructed from uniform sampling (same total # of samples)
   mbar_same_total_samples = MBAR(u_kn_same_total_samples, state_counts_same_total_samples, verbose=False, relative_tolerance=1e-12)
#  Get the 'weights', or reweighted mixture distribution
   weights = mbar.getWeights()
   weights_same_total_samples = mbar_same_total_samples.getWeights()
# Store the weights for later analysis 
   weights_for_each_num_states.extend([weights])
   weights_for_each_num_states_same_total_samples.extend([weights_same_total_samples])

#############
#
# 6) Calculate dimensionless free energies with MBAR
#
#############

# Get the dimensionless free energy differences, and uncertainties in their values
   free_energies,uncertainty_free_energies = mbar.getFreeEnergyDifferences()[0],mbar.getFreeEnergyDifferences()[1]
# Save the free energies
   free_energies_for_each_num_states.extend([free_energies])
# Save the uncertainty in the free energy
   uncertainties_for_each_num_states.extend([uncertainty_free_energies])
# Get the dimensionless free energy differences and uncertainties for the uniform sampling approach
   free_energies,uncertainty_free_energies = mbar_same_total_samples.getFreeEnergyDifferences()[0],mbar_same_total_samples.getFreeEnergyDifferences()[1]
# Save the data
   free_energies_for_each_num_states_same_total_samples.extend([free_energies])
   uncertainties_for_each_num_states_same_total_samples.extend([uncertainty_free_energies])

#############
#
# 7) Calculate < R_Na-Cl > with MBAR
#
#############

# Get < R_Na-Cl >, and uncertainty, with MBAR
   averages, uncertainty = mbar.computeExpectations(distances)[0],mbar.computeExpectations(distances)[1]
# Save < R_Na-Cl >, and uncertainty
   distance_averages_for_each_num_states.extend([averages]) 
   uncertainty_distances_for_each_num_states.extend([uncertainty])
# Get < R_Na-Cl >, and uncertainty, with uniform sampling
   averages, uncertainty = mbar_same_total_samples.computeExpectations(distances_same_total_samples)[0],mbar_same_total_samples.computeExpectations(distances_same_total_samples)[1]
# Save the data
   distance_averages_for_each_num_states_same_total_samples.extend([averages])
   uncertainty_distances_for_each_num_states_same_total_samples.extend([uncertainty])

#############
#
# 8) Analyze uncertainty in < R_Na-Cl > with MBAR as a function of ensemble size
#
#############

   average_uncertainty_for_each_fraction_of_samples[fraction_index][num_states_index] = sum(uncertainty)/len(uncertainty)
   average_distance_for_each_fraction_of_samples[fraction_index][num_states_index] = sum(averages)/len(averages)

   num_states_index = num_states_index + 1

#############
#
# END ITERATION OVER EACH NUMBER OF STATES
#
#############

  fraction_index = fraction_index + 1

if compress_simulation_data: zip_files(output_dir,str(os.getcwd()+"/output.zip"))
#############
#
# END ITERATION OVER FRACTION INDEX (ENSEMBLE SIZE)
#
#############

#############
#
# 9) Plot results for T = 400 K
#
#############

figure_index = 0

# First re-type distance and temperature arrays for pyplot:
distances = np.array([[distances[index]-distances[0] for index in range(0,len(distances))] for distances in distance_averages_for_each_num_states])
temperatures = np.array([[Temp[index] for index in range(0,len(Temp))] for Temp in temperatures_for_each_num_states])
distance_uncertainty = np.array([[uncertainty[index] for index in range(0,len(uncertainty))] for uncertainty in uncertainty_distances_for_each_num_states])

# Define temporary arrays to store the MBAR-evaluated, re-weighted properties at T = 400 K
states_temp = []
distances_temp = []
uncertainty_temp = []
for state_index in range(0,len(temperatures)):
# If the temperature distribution for 'num_states' includes a temperature within 'tol' (10.0 K) of 400 K, we will include it in our analysis, otherwise we omit the dataset
#
# For example, if num_states = 4, the temperatures we sample are: 300.0, 366.6, 433.3, and 500.0.  Thus, we won't plot the results for 'num_states' = 4
 if any(util.isclose(temperatures[state_index][temperature_index],400.0,tol=10.0) for temperature_index in range(0,len(temperatures[state_index]))):
  for temperature_index in range(0,len(temperatures[state_index])):
   if temperatures[state_index][temperature_index] == float(400.0):
    states_temp.append(len(temperatures[state_index]))
    distances_temp.append(distance_averages_for_each_num_states[state_index][temperature_index])
    uncertainty_temp.append(uncertainty_distances_for_each_num_states[state_index][temperature_index])
    break
# Define arrays with the datapoints we will be plotting
states_400 = np.array([states_temp[index] for index in range(0,len(states_temp))])
uncertainty_400 = np.array([uncertainty_temp[index] for index in range(0,len(uncertainty_temp))])
distances_400 = np.array([distances_temp[index] for index in range(0,len(distances_temp))])

#Plot <R_Na-Cl> @ 400 K vs. # of thermodynamic states
figure = pyplot.figure(figure_index)
pyplot.plot(states_400,distances_400,figure=figure)
pyplot.xlabel("Number of thermodynamic states")
pyplot.ylabel("<R_Na-Cl> (Angstroms)")
pyplot.title("Predicted <R_Na-Cl> at 400 K")
pyplot.savefig(str(figures_dir+str("/R_Na-Cl_v_num_states.png")))
#pyplot.show()
pyplot.close()
figure_index = figure_index + 1

#Plot <R_Na-Cl> @ 400 K vs. # of thermodynamic states
figure = pyplot.figure(figure_index)
pyplot.plot(states_400,uncertainty_400,figure=figure)
pyplot.xlabel("Number of thermodynamic states")
pyplot.ylabel("Uncertainty(<R_Na-Cl>) (Angstroms)")
pyplot.title("Uncertainty in <R_Na-Cl> at 400 K")
pyplot.savefig(str(figures_dir+str("/Uncertainty_R_Na-Cl_v_num_states.png")))
#pyplot.show()
pyplot.close()
figure_index = figure_index + 1

# Now we prepare arrays to plot the results with a constant # of samples
states_temp = []
distances_temp = []
uncertainty_temp = []
# Combine data for T = 400.0 K
for state_index in range(0,len(temperatures)):
 if any(util.isclose(temperatures[state_index][temperature_index],400.0,tol=0.001) for temperature_index in range(0,len(temperatures[state_index]))):
  for temperature_index in range(0,len(temperatures[state_index])):
   if temperatures[state_index][temperature_index] == float(400.0):
    states_temp.append(len(temperatures[state_index]))
    distances_temp.append(distance_averages_for_each_num_states_same_total_samples[state_index][temperature_index])
    uncertainty_temp.append(uncertainty_distances_for_each_num_states_same_total_samples[state_index][temperature_index])
    break
states_400 = np.array([states_temp[index] for index in range(0,len(states_temp))])
uncertainty_400 = np.array([uncertainty_temp[index] for index in range(0,len(uncertainty_temp))])
distances_400 = np.array([distances_temp[index] for index in range(0,len(distances_temp))])


#Plot <R_Na-Cl> @ 400 K vs. # of thermodynamic states with constant # of samples
figure = pyplot.figure(figure_index)
pyplot.plot(states_400,distances_400,figure=figure)
pyplot.xlabel("Thermodynamic states")
pyplot.ylabel("<R_Na-Cl> (Angstroms)")
pyplot.title("<R_Na-Cl> (with constant #samples) at 400 K")
pyplot.savefig(str(figures_dir+str("/R_Na-Cl_v_num_states_constant_samples.png")))
#pyplot.show()
pyplot.close()
figure_index = figure_index + 1

#Plot <R_Na-Cl> @ 400 K vs. # of thermodynamic states with constant # of samples
figure = pyplot.figure(figure_index)
pyplot.plot(states_400,uncertainty_400,figure=figure)
pyplot.xlabel("Thermodynamic states")
pyplot.ylabel("Uncertainty(<R_Na-Cl>) (Angstroms)")
pyplot.title("Uncertainty in <R_Na-Cl> with constant # samples at 400 K")
pyplot.savefig(str(figures_dir+str("/Uncertainty_R_Na-Cl_v_num_states_constant_samples.png")))
#pyplot.show()
pyplot.close()
figure_index = figure_index + 1

uncertainty = np.array([average_uncertainty_for_each_fraction_of_samples[index] for index in range(0,len(average_uncertainty_for_each_fraction_of_samples))])
distance = np.array([average_distance_for_each_fraction_of_samples[index] for index in range(0,len(average_distance_for_each_fraction_of_samples))])
samples = np.array([int(round(sub_sampling_level * total_samples)) for sub_sampling_level in sub_sampling_levels])

#Plot <R_Na-Cl> @ 400 K vs. # of thermodynamic states for range of total samples
figure = pyplot.figure(figure_index)
pyplot.plot(states_400,uncertainty,figure=figure)
pyplot.xlabel("Total samples")
pyplot.ylabel("Uncertainty(<R_Na-Cl>) (Angstroms)")
pyplot.title("Uncertainty in <R_Na-Cl> vs. Total samples at 400 K")
pyplot.savefig(str(figures_dir+str("/Uncertainty_R_Na-Cl_v_num_samples_constant_states.png")))
pyplot.legend(samples)
pyplot.show()
pyplot.close()
figure_index = figure_index + 1

#Plot uncertainty in < R_Na-Cl > @ 400 K vs. # of thermodynamic states for range of total samples
figure = pyplot.figure(figure_index)
pyplot.plot(states_400,distance,figure=figure)
pyplot.xlabel("Total samples")
pyplot.ylabel("<R_Na-Cl> (Angstroms)")
pyplot.title("<R_Na-Cl> vs. Total samples at 400 K")
pyplot.savefig(str(figures_dir+str("/Uncertainty_R_Na-Cl_v_num_samples_constant_states.png")))
pyplot.legend(samples)
pyplot.show()
pyplot.close()
figure_index = figure_index + 1


exit() 
