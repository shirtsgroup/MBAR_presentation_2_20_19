import numpy as np
import math

def get_distances(pdb_file,simulation_steps):
 distances = np.array([0. for distance in range(0,simulation_steps)])
 output_obj = open(pdb_file,'r')
 step = 0
 Na_coord = []
 Cl_coord = []
 for line in output_obj.readlines():
   if line.split(' ')[0] == "HETATM":
    if line.split(' ')[5] == "Na+":
     Na_coord = np.array([float(line.split(' ')[18]),float(line.split(' ')[21]),float(line.split(' ')[24])])
    if line.split(' ')[5] == "Cl-":
     Cl_coord = np.array([float(line.split(' ')[18]),float(line.split(' ')[21]),float(line.split(' ')[24])])

   if Na_coord != [] and Cl_coord != []:
    sum_term = sum([(Na_coord[index]-Cl_coord[index])**2 for index in range(0,3)])
    distances[step] = math.sqrt(sum_term)
    Na_coord = []
    Cl_coord = []
    step = step + 1
 output_obj.close()
 return(distances)

def isclose(a, b, tol):
        return abs(a-b) <= tol
