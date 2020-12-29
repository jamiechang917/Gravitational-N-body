import numpy as np
import numba as nb
from numba import cuda, prange
import math

import time
import csv

import cProfile

#===========Problems need to be fixed=============#

#==============Simulation Parameters==============#
N = 5000
dt = 0.01
t_max = 20
softening_factor = 0.1
#==============Physical Parameters================#
mass = 1
avg_vel  = 6
G = 1
#==============Environment Parameters=============#
r_range = 20 # the max range of the distance from origin to initial position of particles
#==============File Parameters====================#
SAVE_DATA = True
CSV_PATH = r"C:\Users\Jamie Chang\Desktop\GalacticSim\Python\\"

#==============Others Parameters==================#
t = 0
time_steps = int(t_max/dt)
filename = r"N_Body_"+time.ctime().replace(" ","_").replace(":","")+".csv"

threadsperblock = (32,32)
blockspergrid_x = math.ceil(N/threadsperblock[0])
blockspergrid_y = math.ceil(N/threadsperblock[1])
blockspergrid = (blockspergrid_x,blockspergrid_y)
#=================================================#

@nb.njit()
def generate_particles(N,r_range,avg_vel): # Initialization
        np.random.seed(2020)
        random_theta = np.random.uniform(0,2*np.pi,(N,1))
        random_phi = np.random.uniform(0,2*np.pi,(N,1))
        #vel_M = avg_vel*np.concatenate((np.sin(random_theta)*np.cos(random_phi),np.sin(random_theta)*np.sin(random_phi),np.cos(random_theta)),axis=1)
        pos_M = np.zeros((N,3))
        i = 0
        while i < N:
            pos = np.random.uniform(-r_range,r_range,3)
            if np.linalg.norm(pos) <= r_range:
                pos_M[i] = pos
                i += 1
        vel_M = np.zeros((N,3))
        for i in range(N):
            unit_vec = np.array([-pos_M[i][1],pos_M[i][0],0])/np.linalg.norm(np.array([-pos_M[i][1],pos_M[i][0],0]))
            vel_M[i] = avg_vel*unit_vec
        return pos_M, vel_M

pos_M, vel_M = generate_particles(N,r_range,avg_vel)

@cuda.jit()
def total_U(pos,U):
    i,j = cuda.grid(2)
    if i < pos.shape[0] and j < pos.shape[0]:
        r_x = pos[i][0] - pos[j][0]
        r_y = pos[i][1] - pos[j][1]
        r_z = pos[i][2] - pos[j][2]
        r_norm = ((r_x**2)+(r_y**2)+(r_z**2))**0.5
        cuda.atomic.add(U,0,-0.5*(G*mass*mass)/r_norm) # multiply 0.5 because it will calculate two times

@nb.njit()
def total_K(vel):
    K = 0
    for i in range(N):
        v = np.float(np.linalg.norm(vel[i]))
        K = K + 0.5*mass*(v**2)
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
        r_norm = ((r_x**2)+(r_y**2)+(r_z**2))**0.5
        if r_norm == 0:
            pass
        elif r_norm <= softening_factor:
            cuda.atomic.add(g,(i,0),-(G*mass*(((r_norm**2) + (softening_factor**2))**-1.5)*r_x))
            cuda.atomic.add(g,(i,1),-(G*mass*(((r_norm**2) + (softening_factor**2))**-1.5)*r_y))
            cuda.atomic.add(g,(i,2),-(G*mass*(((r_norm**2) + (softening_factor**2))**-1.5)*r_z))
        else:
            cuda.atomic.add(g,(i,0),-(G*mass*((r_norm**-3)*r_x)))
            cuda.atomic.add(g,(i,1),-(G*mass*((r_norm**-3)*r_y)))
            cuda.atomic.add(g,(i,2),-(G*mass*((r_norm**-3)*r_z)))

@nb.njit()
def brute_force_method_elastic_collision(pos,vel):
    g = np.zeros((N,3)) # 3 for 3D
    for i in range(N-1):
        for j in range(i+1,N):
            r = pos[i] - pos[j]
            r_norm = np.linalg.norm(r)
            if r_norm <= softening_factor:
                #print(vel[i],vel[j],"BEFORE")
                #!!!!!Important, to be careful for this "swapping" part
                vel_copy = vel.copy()
                vel[i] = vel_copy[j]
                vel[j] = vel_copy[i]
                #print(vel[i],vel[j],"AFTER")
            else:
                # g[i] = g[i] -(G*mass*(((r_norm**2) + (softening_factor**2))**-1.5)*r)
                # g[j] = g[j] +(G*mass*(((r_norm**2) + (softening_factor**2))**-1.5)*r)
                g[i] = g[i] -(G*mass*((r_norm**-3)*r))
                g[j] = g[j] +(G*mass*((r_norm**-3)*r))
    return g,vel

@nb.njit()
def update_elastic_collision(pos,vel):
    result = brute_force_method_elastic_collision(pos=pos,vel=vel)
    vel_next = result[1] + result[0]*dt #-------------------------------2
    pos_next = pos + vel_next*dt  #-------------------------------3
    return pos_next,vel_next

def update_leapfrog_cuda(pos,vel):
    threadsperblock = (32,32)
    blockspergrid_x = math.ceil(N/threadsperblock[0])
    blockspergrid_y = math.ceil(N/threadsperblock[1])
    blockspergrid = (blockspergrid_x,blockspergrid_y)

    g0 = np.zeros((N,3))
    brute_force_method_cuda[blockspergrid,threadsperblock](g0,pos)
    vel_half = vel + 0.5*dt*g0
    pos_next = pos + vel_half*dt
    g_next = np.zeros((N,3))
    brute_force_method_cuda[blockspergrid,threadsperblock](g_next,pos_next)
    vel_next = vel_half + 0.5*dt*g_next
    return pos_next,vel_next


def datasaver(init=False,t=0):
    if init == True: # Write all parameters in first row.
        with open(CSV_PATH+filename,"a",newline="") as csvfile: 
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['Parameters',N,dt,t_max,softening_factor,mass,avg_vel,G,r_range])
    else:
        with open(CSV_PATH+filename,"a",newline="") as csvfile: 
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([t])
            writer.writerow(pos_M.flatten())
            writer.writerow(vel_M.flatten())

def main():
    global pos_M ,vel_M
    if SAVE_DATA == True:
        datasaver(init=True)
    counter = 0
    print("CM: ",CM(pos_M))
    for time in np.linspace(0,t_max,time_steps):
        if SAVE_DATA == True:
            datasaver(t=time)
        pos_M,vel_M = update_leapfrog_cuda(pos=pos_M,vel=vel_M)
        if counter == 100 or time==t_max:
            counter = 0
            U = np.array([0.])
            total_U[blockspergrid,threadsperblock](pos_M,U)
            E = total_K(vel_M) + U[0]
            #print("CM: ",CM(pos_M))
            print("E: %d, E0: %d, Error: %0.3f %%" % (E,E0,100*(E-E0)/E0))
        print("Recording the data. Progress:%0.2f %%" % (100*time/t_max))
        counter += 1
        
if __name__ == '__main__':
    program_start = time.time()
    #cProfile.run('main()',sort="cumulative")
    main()
    program_end = time.time()
    print(f"N={N}, Total computation time: {program_end-program_start}s, filename: {filename}")