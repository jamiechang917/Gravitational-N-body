import numpy as np
import numba as nb
from numba import cuda, prange
import math

import time
import csv

import cProfile

import diskgalaxy
#===========Problems need to be fixed=============#

#==============Simulation Parameters==============#
N = 10000
dt = 0.1
t_max = 150 # 1s equal ~2.0Myr
softening_factor = 0.1
COLLISION_MODE = False
#==============Physical Parameters================#
solar_mass = np.float(1.9*(10**30)) #kg
kpc = np.float(3.08568*(10**19)) #m
Myr = np.float(3.1557*(10**13)) #s
M_galaxy = np.float(5.6*(10**10)) # solar mass (disk mass)

mass = 1.0/N
G = 1.0
a = 2.1 #kpc # https://arxiv.org/pdf/astro-ph/9710197.pdf (a=2.1,b=0.21,R_max=40,z_max=3,space_split=30)
b = 0.21 #kpc
Q = 0.8

# For second galaxy
G2_M = 0.2*M_galaxy
G2_N = int(N*G2_M/M_galaxy)
G2_a = 1
G2_b = 1
G2_Q = 1.2
G2_R_max = 5
G2_z_max = 5
G2_space_split = 10
G2_displacement = np.array([[0,0,5]]) # kpc
G2_speed = np.array([[0,0,-0.1]]) # kpc/timescale

#==============Environment Parameters=============#
R_max = 40 #40
z_max = 3 #3
space_split = 30
#==============File Parameters====================#
SAVE_DATA = True
CSV_PATH = r"C:\Users\Jamie Chang\Desktop\GalacticSim\Python\\"
#==============Others Parameters==================#
t = 0.0
time_steps = int(t_max/dt)
filename = r"N_Body_"+time.ctime().replace(" ","_").replace(":","")+".csv"

threadsperblock = (32,32)
blockspergrid_x = math.ceil(N/threadsperblock[0])
blockspergrid_y = math.ceil(N/threadsperblock[1])
blockspergrid = (blockspergrid_x,blockspergrid_y)
#avg_vel  = 0.0
#r_range = 20 # the max range of the distance from origin to initial position of particles
#=================================================#
# @nb.njit()
# def generate_particles(N,r_range,avg_vel): # Initialization
#     np.random.seed(2020)
#     random_theta = np.random.uniform(0,2*np.pi,(N,1))
#     random_phi = np.random.uniform(0,2*np.pi,(N,1))
#     #vel_M = avg_vel*np.concatenate((np.sin(random_theta)*np.cos(random_phi),np.sin(random_theta)*np.sin(random_phi),np.cos(random_theta)),axis=1)
#     pos_M = np.zeros((N,3))
#     i = 0
#     while i < N:
#         pos = np.random.uniform(-r_range,r_range,3)
#         if np.linalg.norm(pos) <= r_range:
#             pos_M[i] = pos
#             i += 1
#     vel_M = np.zeros((N,3))
#     for i in range(N):
#         unit_vec = np.array([-pos_M[i][1],pos_M[i][0],0])/np.linalg.norm(np.array([-pos_M[i][1],pos_M[i][0],0]))
#         vel_M[i] = avg_vel*unit_vec
#     return pos_M, vel_M

if COLLISION_MODE == True:
    G1_pos , G1_vel =  diskgalaxy.generate_particles(N=N,m=mass,a=a,b=b,R_max=R_max,z_max=z_max,space_split=space_split,G=G,Q=Q)
    G2_pos , G2_vel = diskgalaxy.generate_particles(N=G2_N,m=mass,a=G2_a,b=G2_b,R_max=G2_R_max,z_max=G2_z_max,space_split=G2_space_split,G=G,Q=G2_Q)
    print("G1 N=",G1_pos.shape[0]," G2 N=",G2_pos.shape[0])
    G1_N = G1_pos.shape[0]
    G2_N = G2_pos.shape[0]
    G2_pos = G2_pos + G2_displacement.repeat([G2_N],axis=0)
    G2_vel = G2_vel + G2_speed.repeat([G2_N],axis=0)

    pos_M = np.concatenate((G1_pos,G2_pos))
    vel_M = np.concatenate((G1_vel,G2_vel))
else:
    pos_M, vel_M = diskgalaxy.generate_particles(N=N,m=mass,a=a,b=b,R_max=R_max,z_max=z_max,space_split=space_split,G=G,Q=Q)

print("Contain %0.2f %% stars, N = %d" % (100*abs(pos_M.shape[0])/N,pos_M.shape[0]))
N = pos_M.shape[0] # The real N must be declared.

@cuda.jit()
def total_U(pos,U):
    i,j = cuda.grid(2)
    if i < pos.shape[0] and j < pos.shape[0]:
        r_x = pos[i][0] - pos[j][0]
        r_y = pos[i][1] - pos[j][1]
        r_z = pos[i][2] - pos[j][2]
        r_norm = (r_x**2+r_y**2+r_z**2+softening_factor**2)**0.5
        cuda.atomic.add(U,0,-0.5*(G*mass*mass)/r_norm) # multiply 0.5 because it will calculate twice

@nb.njit()
def total_K(vel):
    K = 0.0
    for i in range(N):
        v = np.float(np.linalg.norm(vel[i]))
        K += 0.5*mass*(v**2)
    return K

@nb.njit()
def CM(pos):
    x = np.sum(pos[:,0])/N
    y = np.sum(pos[:,1])/N
    z = np.sum(pos[:,2])/N
    return (x,y,z)

U0 = np.array([0.])
total_U[blockspergrid,threadsperblock](pos_M,U0)
E0 = total_K(vel_M) + U0[0]

@cuda.jit()
def brute_force_method_cuda(g,pos):
    i,j = cuda.grid(2)  # (i,j) range from (0,0) to (N-1,N-1)
    if i < pos.shape[0] and j < pos.shape[0]: # it will calculate the g when i,j less than N ((i,j) range from (0,0) to (N-1,N-1))
        r_x = pos[i][0] - pos[j][0]
        r_y = pos[i][1] - pos[j][1]
        r_z = pos[i][2] - pos[j][2]
        inv_3 = (r_x**2+r_y**2+r_z**2+softening_factor**2)**(-1.5)
        if i == j:
            pass
        else:
            cuda.atomic.add(g,(i,0),-G*mass*inv_3*r_x)
            cuda.atomic.add(g,(i,1),-G*mass*inv_3*r_y)
            cuda.atomic.add(g,(i,2),-G*mass*inv_3*r_z)

def update_leapfrog_cuda(pos,vel):
    threadsperblock = (32,32)
    blockspergrid_x = math.ceil(N/threadsperblock[0])
    blockspergrid_y = math.ceil(N/threadsperblock[1])
    blockspergrid = (blockspergrid_x,blockspergrid_y)

    g0 = np.zeros((N,3))
    brute_force_method_cuda[blockspergrid,threadsperblock](g0,pos)
    vel = vel + 0.5*g0*dt
    pos = pos + vel*dt
    g_next = np.zeros((N,3))
    brute_force_method_cuda[blockspergrid,threadsperblock](g_next,pos)
    vel = vel + 0.5*g_next*dt
    return pos,vel

def datasaver(init=False,record_energy=False,t=t,K=0,U=0,E=0):
    if init == True: # Write all parameters in first row.
        with open(CSV_PATH+filename,"a",newline="") as csvfile: 
            writer = csv.writer(csvfile, delimiter=',')
            if COLLISION_MODE == True:
                writer.writerow(['Parameters',N,G1_N,G2_N,dt,t_max,softening_factor,mass,G,a,G2_a,b,G2_b,Q,G2_Q,M_galaxy,G2_M,R_max,G2_R_max,z_max,G2_z_max,space_split,G2_space_split,G2_displacement,G2_speed])
            else:
                writer.writerow(['Parameters',N,dt,t_max,softening_factor,mass,G,a,b,Q,M_galaxy,R_max,z_max,space_split])
                
    else:
        with open(CSV_PATH+filename,"a",newline="") as csvfile: 
            writer = csv.writer(csvfile, delimiter=',')
            if record_energy == True:
                writer.writerow([t,K,U,E])
            else:
                writer.writerow([t])
            writer.writerow(pos_M.flatten())
            writer.writerow(vel_M.flatten())

def main():
    global pos_M ,vel_M
    if SAVE_DATA == True:
        datasaver(init=True)
    RECORD_ENERGY = False
    counter = 0
    # print("CM: ",CM(pos_M))
    for time in np.linspace(0,t_max,time_steps):
        if counter%10 == 0 or time==t_max:
            U = np.array([0.])
            total_U[blockspergrid,threadsperblock](pos_M,U)
            K = total_K(vel_M)
            E = K + U[0]
            #print("CM: ",CM(pos_M))
            print("E: %0.2f, E0: %0.2f, Error: %0.3f %%" % (E,E0,100*(E-E0)/E0))
            if SAVE_DATA == True:
                datasaver(t=time,record_energy=True,K=K,U=U[0],E=E)
        elif SAVE_DATA == True:
            datasaver(t=time)
        pos_M,vel_M = update_leapfrog_cuda(pos=pos_M,vel=vel_M)
        print("Recording the data. Progress:%0.2f %%" % (100*time/t_max))
        counter += 1

        
if __name__ == '__main__':
    program_start = time.time()
    #cProfile.run('main()',sort="cumulative")
    main()
    program_end = time.time()
    print(f"N={N}, Total computation time: {program_end-program_start}s, filename: {filename}")