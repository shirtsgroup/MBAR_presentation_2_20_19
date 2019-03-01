# This Python script contains example NaCl simulations in OpenMM, 
# followed by analysis with MBAR, and WHAM.

# The protocol is broken up into the following sections:

# 1) Run-time options
# 2) Prepare and run OpenMM simulations (Langevin dynamics @ named temps.)
# 3) Sub-sample simulation data to obtain decorrelated samples
# 4) Calculate 'weights' for each thermodynamic state with MBAR
# 5) Calculate dimensionless free energies with MBAR

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
from multiprocessing import Pool
from openmmtools.testsystems import SodiumChlorideCrystal
from pymbar import MBAR, timeseries

print("Running the MBAR free energy protocol for sodium chloride...")

#############
#
# 1) Run-time options
#
#############

processors = 18
simulation_time_step = 0.002 # Units = picoseconds
kB = 0.008314462  #Boltzmann constant (Gas constant) in kJ/(mol*K)
state_range = [num_states for num_states in range(2,50)]
min_temp = 300.0 # Units = Kelvin
max_temp = 500.0 # Units = Kelvin
if cwd().split('/')[-1] == 'FreeEnergy':
 output_dir = 'output/'
 figures_dir = 'figures/'
if cwd().split('/')[-1] == 'MBAR_presentation_2_20_19':
 output_dir = 'NaCl/FreeEnergy/output/'
 figures_dir = 'NaCl/FreeEnergy/figures/'
simulation_steps = 100000
print_frequency = 10 # Number of steps to skip when printing output
nskip = 10 # Number of steps to skip when reading timeseries data to find the equilibration time
simulation_data_exists = True
analysis_data_exists = True

#############
#
# 2) Prepare and run OpenMM simulation
#
#############

# An OpenMM simulation requires three input objects: a 'system', an 'integrator', and a 'context'

total_simulation_time = simulation_time_step * simulation_steps # Units = picoseconds
# Define arrays that will store our data
free_energies_for_each_num_states = []
free_energies_for_each_num_states_same_total_samples = []
uncertainties_for_each_num_states = []
uncertainties_for_each_num_states_same_total_samples = []
distances_for_each_num_states = []
distances_for_each_num_states_same_total_samples = []
distance_averages_for_each_num_states = []
distance_averages_for_each_num_states_same_total_samples = []
uncertainty_distances_for_each_num_states = []
uncertainty_distances_for_each_num_states_same_total_samples = []
weights_for_each_num_states = []
weights_for_each_num_states_same_total_samples = []
temperatures_for_each_num_states = []
average_uncertainty_for_each_fraction_of_samples = [[0.0 for num_states in state_range] for index in range(0.5,0.8,0.05)]
average_distance_for_each_fraction_of_samples = [[0.0 for num_states in state_range] for index in range(0.5,0.8,0.05)]


def individual_simulation_procedure(temperature):
   system = SodiumChlorideCrystal() # Define a system
   integrator = LangevinIntegrator(temperature, total_simulation_time, simulation_time_step) # Define an integrator
   simulation = Simulation(system.topology, system.system, integrator) # Define a simulation 'context'
   if not(os.path.exists(str(output_dir+str(temperature)))): os.mkdir(str(output_dir+str(temperature)))
   simulation.reporters.append(PDBReporter(str(output_dir+str(temperature)+"/coordinates.pdb"),1))
   simulation.reporters.append(StateDataReporter(str(output_dir+str(temperature)+"/sim_data.dat"), print_frequency, \
   step=True, totalEnergy=True, potentialEnergy=True, kineticEnergy=True, temperature=True))
   simulation.context.setPositions(system.positions) # Assign particle positions for this context
   simulation.minimizeEnergy() # Set the simulation type to energy minimization
   simulation.step(simulation_steps) # Run the simulation
   return

# If we have run simulations, but we want to re-analyze the output, we start here

def individual_analysis_procedure(temperature):
#  Read in the total energies
   U_uncorrelated = np.array([])
   U_reduced = np.array([])
   output_obj = open(str(output_dir+str(temperature)+"/sim_data.dat"),'r')
# E_total_all stores the total energies from NaCl simulation output
   E_total_all_temp = np.array([l.split(',')[3] for l in output_obj.readlines()])
   output_obj.close()
# Read in the distances
   distances = util.get_distances(str(output_dir+str(temperature)+"/coordinates.pdb"),simulation_steps)
   E_total_all = np.array(np.delete(E_total_all_temp,0,0),dtype=float)
   [t0, g, Neff_max] = timeseries.detectEquilibration(E_total_all,nskip=nskip)
   E_total_equil = E_total_all[t0:]
   uncorrelated_energy_indices = timeseries.subsampleCorrelatedData(E_total_equil, g=g) # Determine indices of uncorrelated samples
   np.savetxt(str(output_dir+str(temperature)+'/uncorrelated_total_energies.dat'),E_total_equil[uncorrelated_energy_indices])
   np.savetxt(str(output_dir+str(temperature)+'/uncorrelated_distances.dat'),distances[uncorrelated_energy_indices])
   U_uncorrelated = np.append(U_uncorrelated,E_total_equil[uncorrelated_energy_indices]) # Uncorrelated total energies
   U_reduced = np.append(U_reduced,np.array([E_total_equil[uncorrelated_energy_indices]/(temperature*kB)]))
   return

def read_simulation_data(temperature):
   U_uncorrelated = np.array([])
   distances_obj = open(str(output_dir+str(temperature)+'/uncorrelated_distances.dat'),'r')
   distances = np.array([float(l) for l in distances_obj.readlines()])
   distances_obj.close()
   uncorrelated_energies_obj = open(str(output_dir+str(temperature)+'/uncorrelated_total_energies.dat'),'r')
   U_uncorrelated = np.append(U_uncorrelated,[float(l) for l in uncorrelated_energies_obj.readlines()])
   uncorrelated_energies_obj.close()
   return(U_uncorrelated,distances)

if __name__ == '__main__':

#############
#
# Begin iteration over states
#
#############

# If we haven't run simulations yet we start here
 if not(simulation_data_exists):
  for num_states in state_range:
   print("Running simulations with "+str(num_states)+" states")
#  Define the temperature range based upon the number of states
   simulation_temperatures = [round((min_temp + i*(max_temp-min_temp)/(num_states-1)),1) for i in range(0,num_states)]
   for temperature in simulation_temperatures:
    if os.path.exists(str(output_dir+str(temperature))):
     simulation_temperatures.remove(temperature)

   if not(simulation_data_exists):
    pool = Pool(processes=processors)
    pool.map(individual_simulation_procedure,simulation_temperatures)

# If we haven't analyzed the simulation data yet:
 if not(analysis_data_exists):
  for num_states in state_range:
   print("Running analysis with "+str(num_states)+" states")
   temperatures = [round((min_temp + i*(max_temp-min_temp)/(num_states-1)),1) for i in range(0,num_states)]
   if not any(os.path.exists(str(output_dir+str(temperatures[index]))) for index in range(0,len(temperatures))):
    pool = Pool(processes=processors)
    pool.map(individual_analysis_procedure,temperatures)

# If we've already performed simulations and analyzed data:
 if analysis_data_exists: print("Reading output from previous analyses...")
 num_states_index = 0

# Define arrays to analyze data with a different number of total samples
 average_uncertainty_for_each_fraction_of_samples = []
 average_distance_for_each_fraction_of_samples = []
 fraction_index
 for fraction_of_samples in range(0.5,0.8,0.5):
  for num_states in state_range:
   print("with "+str(num_states)+" states...")
# Create an array to store the number of counts in each state
   state_counts = []
   state_counts_same_total_samples = []
   U_uncorrelated = []
   U_uncorrelated_same_total_samples = []
   distances = []
   distances_same_total_samples = []
  
   temperatures = [round((min_temp + i*(max_temp-min_temp)/(num_states-1)),1) for i in range(0,num_states)]
#  print(temperatures)
   temperatures_for_each_num_states.extend([temperatures])
   if num_states == 2:
    total_samples = int(round(2.0*fraction_of_samples*min(len(read_simulation_data(temperatures[0])[0]),len(read_simulation_data(temperatures[0])[0]))))
    print("Total samples ="+str(total_samples))
   samples_per_state = int(round(total_samples/num_states)-1)
   for temperature in temperatures:
    U_uncorrelated_temp, distances_temp = read_simulation_data(temperature) 
    U_uncorrelated.extend(U_uncorrelated_temp)
    sample_indices = []
#   print(samples_per_state)
    while len(sample_indices) < samples_per_state:
#    print(len(sample_indices))
     sample_index = random.randint(0,len(U_uncorrelated_temp)-1)
     if sample_index not in sample_indices: 
      sample_indices.extend([int(sample_index)])
#     print(sample_index)
#     print(U_uncorrelated_temp[sample_index])
      U_uncorrelated_same_total_samples.extend([U_uncorrelated_temp[sample_index]])
      distances_same_total_samples.extend([distances_temp[sample_index]])
    state_counts.extend([len(U_uncorrelated_temp)])
    state_counts_same_total_samples.extend([samples_per_state])
    distances.extend(distances_temp)
#   distances_same_total_samples.extend([distances_temp[sample] for sample in sample_indices])
#############
#
# 4) Calculate 'weights' for each thermodynamic state with MBAR
#
#############

# 'u_kn' contains the decorrelated reduced potential energies evaluated at each temperature (state)
   u_kn = np.array([[float(float(U_uncorrelated[sample])/float(temperature)) for sample in range(0,len(U_uncorrelated))] for temperature in temperatures])
   u_kn_same_total_samples = np.array([[float(float(U_uncorrelated_same_total_samples[sample])/float(temperature)) for sample in range(0,len(U_uncorrelated_same_total_samples))] for temperature in temperatures])
#  print(u_kn_same_total_samples)
#  exit()
# Initialize MBAR
   mbar = MBAR(u_kn, state_counts, verbose=False, relative_tolerance=1e-12)
   mbar_same_total_samples = MBAR(u_kn_same_total_samples, state_counts_same_total_samples, verbose=False, relative_tolerance=1e-12)
#  Get the 'weights', or reweighted mixture distribution
   weights = mbar.getWeights()
   weights_same_total_samples = mbar_same_total_samples.getWeights()
# Store the weights 
   weights_for_each_num_states.extend([weights])
   weights_for_each_num_states_same_total_samples.extend([weights_same_total_samples])

#############
#
# 5) Calculate dimensionless free energies and <R_Na-Cl> with MBAR
#
#############

# Get the dimensionless free energy differences
   free_energies,uncertainty_free_energies = mbar.getFreeEnergyDifferences()[0],mbar.getFreeEnergyDifferences()[1]
# Save the free energies for this number of states for comparison plots later on
   free_energies_for_each_num_states.extend([free_energies])
   uncertainties_for_each_num_states.extend([uncertainty_free_energies])
   free_energies,uncertainty_free_energies = mbar_same_total_samples.getFreeEnergyDifferences()[0],mbar_same_total_samples.getFreeEnergyDifferences()[1]
   free_energies_for_each_num_states_same_total_samples.extend([free_energies])
   uncertainties_for_each_num_states_same_total_samples.extend([uncertainty_free_energies])

# Get the expectation value of the Na-Cl distance
   averages, uncertainty = mbar.computeExpectations(distances)[0],mbar.computeExpectations(distances)[1]
   distance_averages_for_each_num_states.extend([averages]) 
   uncertainty_distances_for_each_num_states.extend([uncertainty])
   averages, uncertainty = mbar_same_total_samples.computeExpectations(distances_same_total_samples)[0],mbar_same_total_samples.computeExpectations(distances_same_total_samples)[1]
   distance_averages_for_each_num_states_same_total_samples.extend([averages])
   uncertainty_distances_for_each_num_states_same_total_samples.extend([uncertainty])

   average_uncertainty_for_each_fraction_of_samples[fraction_index][num_states_index] = sum(uncertainty)/len(uncertainty)
   average_distance_for_each_fraction_of_samples[fraction_index][num_states_index] = sum(averages)/len(averages)

#  print(uncertainty)
   num_states_index = num_states_index + 1
  fraction_index = fraction_index + 1



#############
#
# END ITERATION OVER EACH NUMBER OF STATES
#
#############


#############
#
# 6) Plot the results
#
#############

figure_index = 0

# Plot distribution functions versus Temp.

# First re-type distance and temperature arrays for pyplot:
distances = np.array([[distances[index]-distances[0] for index in range(0,len(distances))] for distances in distance_averages_for_each_num_states])
temperatures = np.array([[Temp[index] for index in range(0,len(Temp))] for Temp in temperatures_for_each_num_states])
distance_uncertainty = np.array([[uncertainty[index] for index in range(0,len(uncertainty))] for uncertainty in uncertainty_distances_for_each_num_states])

states_temp = []
distances_temp = []
uncertainty_temp = []
# Combine data for T = 400.0 K
for state_index in range(0,len(temperatures)):
 if any(util.isclose(temperatures[state_index][temperature_index],400.0,tol=0.001) for temperature_index in range(0,len(temperatures[state_index]))):
  for temperature_index in range(0,len(temperatures[state_index])):
   if temperatures[state_index][temperature_index] == float(400.0):
    states_temp.append(len(temperatures[state_index]))
    distances_temp.append(distance_averages_for_each_num_states[state_index][temperature_index])
    uncertainty_temp.append(uncertainty_distances_for_each_num_states[state_index][temperature_index])
    break
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
pyplot.show()
pyplot.close()
figure_index = figure_index + 1

#Plot <R_Na-Cl> @ 400 K vs. # of thermodynamic states
figure = pyplot.figure(figure_index)
pyplot.plot(states_400,uncertainty_400,figure=figure)
pyplot.xlabel("Number of thermodynamic states")
pyplot.ylabel("Uncertainty(<R_Na-Cl>) (Angstroms)")
pyplot.title("Uncertainty in <R_Na-Cl> at 400 K")
pyplot.savefig(str(figures_dir+str("/Uncertainty_R_Na-Cl_v_num_states.png")))
pyplot.show()
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
pyplot.xlabel("Number of thermodynamic states")
pyplot.ylabel("<R_Na-Cl> (Angstroms)")
pyplot.title("Predicted <R_Na-Cl> (with constant #samples) at 400 K")
pyplot.savefig(str(figures_dir+str("/R_Na-Cl_v_num_states_constant_samples.png")))
pyplot.show()
pyplot.close()
figure_index = figure_index + 1

#Plot <R_Na-Cl> @ 400 K vs. # of thermodynamic states with constant # of samples
figure = pyplot.figure(figure_index)
pyplot.plot(states_400,uncertainty_400,figure=figure)
pyplot.xlabel("Number of thermodynamic states")
pyplot.ylabel("Uncertainty(<R_Na-Cl>) (Angstroms)")
pyplot.title("Uncertainty in <R_Na-Cl> (with constant #samples) at 400 K")
pyplot.savefig(str(figures_dir+str("/Uncertainty_R_Na-Cl_v_num_states_constant_samples.png")))
pyplot.show()
pyplot.close()
figure_index = figure_index + 1

#Script works until here

uncertainty_400 = np.array([uncertainty for index in range(0,len(uncertainty_temp))])
distances_400 = np.array([distances_temp[index] for index in range(0,len(distances_temp))])

#Plot <R_Na-Cl> @ 400 K vs. # of thermodynamic states for different numbers of total samples
figure = pyplot.figure(figure_index)
pyplot.plot(states_400,uncertainty_400,figure=figure)
pyplot.xlabel("Number of thermodynamic states")
pyplot.ylabel("Uncertainty(<R_Na-Cl>) (Angstroms)")
pyplot.title("Uncertainty in <R_Na-Cl> (with constant #samples) at 400 K")
pyplot.savefig(str(figures_dir+str("/Uncertainty_R_Na-Cl_v_num_states_constant_samples.png")))
pyplot.show()
pyplot.close()
figure_index = figure_index + 1

exit() 
