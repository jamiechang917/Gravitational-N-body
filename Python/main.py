import numpy as np
import numba as nb
from numba import cuda, prange

import time
import csv

import cProfile

# Updates Notes
# Add function to output/read csv data.
# Remove animation

#===========Problems need to be fixed=============#
# 1. Particles.generate_particles() needs to be improved
# 2. Brute_force_method() needs to be improved from N^2 to 0.5*N!/N-2!

#==============Simulation Parameters==============#
N = 1000
dt = 0.01
t_max = 5
softening_factor = 0.5
#==============Physical Parameters================#
mass = 1
avg_vel  = 0
G = 1
#==============Environment Parameters=============#
#x_range = [-50,50] # range of initial position of particles in x axis
#y_range = [-50,50] # range of initial position of particles in y axis
r_range = 50 # the max range of the distance from origin to initial position of particles
#box_length = 30
#==============Videos Parameters==================#
FPS = 30
VIDEO_PATH = r"C:\Users\Jamie Chang\Desktop\GalacticSim\Python\videos"
CSV_PATH = r"C:\Users\Jamie Chang\Desktop\GalacticSim\Python"

#==============Others Parameters==================#
t = 0
time_steps = int(t_max/dt)
filename = r"\N_Body_"+time.ctime().replace(" ","_").replace(":","")+".csv"
#=================================================#

def generate_particles(N,r_range,avg_vel): # Initialization
        np.random.seed(2020)
        random_theta = np.random.uniform(0,2*np.pi,(N,1))
        pos_M = np.random.uniform(0,r_range,(N,1)) * np.concatenate((np.cos(random_theta),np.sin(random_theta)),axis=1)
        vel_M = avg_vel*np.concatenate((np.cos(random_theta),np.sin(random_theta)),axis=1)
        return pos_M, vel_M

pos_M, vel_M = generate_particles(N,r_range,avg_vel)

@nb.njit
def total_U(pos):
    U = 0
    for i in range(N-1):
        for j in range(i+1,N):
            r = np.float(np.linalg.norm(pos[i]-pos[j]))
            U = U - np.divide(G*(mass**2),r)
    return U

@nb.njit
def total_K(vel):
    K = 0
    for i in range(N):
        v = np.float(np.linalg.norm(vel[i]))
        K = K + 0.5*mass*(v**2)
    return K

E0 = total_K(vel_M) + total_U(pos_M)

# @nb.njit()
# def brute_force_method(pos):
#     g = np.zeros((N,2)) # 2 for 2D
#     for i in range(N):
#         for j in range(N):
#             r = pos[i] - pos[j]
#             r_norm = np.linalg.norm(r)
#             if r_norm == 0.0:
#                 continue
#             elif r_norm <= softening_factor:
#                 g[i] = g[i] -(G*mass*((softening_factor)**-2)*(np.divide(r,r_norm)))
#             else:
#                 g[i] = g[i] -(G*mass*(((r_norm**2) + (softening_factor**2))**-1.5)*r) #-------------------------------1
#     #print('g\n',g)
#     return g

@nb.njit()
def brute_force_method(pos):
    g = np.zeros((N,2)) # 2 for 2D
    for i in range(N-1):
        for j in range(i+1,N):
            r = pos[i] - pos[j]
            r_norm = np.linalg.norm(r)
            if r_norm <= softening_factor:
                g[i] = g[i] -(G*mass*(((r_norm**2) + (softening_factor**2))**-1.5)*r)
                g[j] = g[j] +(G*mass*(((r_norm**2) + (softening_factor**2))**-1.5)*r)
            else:
                g[i] = g[i] -(G*mass*(r_norm**-3)*r) #-------------------------------1
                g[j] = g[j] +(G*mass*(r_norm**-3)*r)
    return g

@nb.njit()
def brute_force_method_elastic_collision(pos,vel):
    g = np.zeros((N,2)) # 2 for 2D
    for i in range(N-1):
        for j in range(i+1,N):
            r = pos[i] - pos[j]
            r_norm = np.linalg.norm(r)
            if r_norm <= softening_factor:
                #print(vel[i],vel[j],"BEFORE")
                #!!!!!Important, to be careful for this "swapping" part
                vel_T = vel.copy()
                vel[i] = vel_T[j]
                vel[j] = vel_T[i]
                #print(vel[i],vel[j],"AFTER")
            else:
                # g[i] = g[i] -(G*mass*(((r_norm**2) + (softening_factor**2))**-1.5)*r)
                # g[j] = g[j] +(G*mass*(((r_norm**2) + (softening_factor**2))**-1.5)*r)
                g[i] = g[i] -(G*mass*((r_norm**-3)*r))
                g[j] = g[j] +(G*mass*((r_norm**-3)*r))
    return g,vel


@nb.njit()
def update(pos,vel):
    vel_next = vel + brute_force_method(pos=pos)*dt  #-------------------------------2
    pos_next = pos + vel*dt  #-------------------------------3
    return pos_next,vel_next

@nb.njit()
def update_elastic_collision(pos,vel):
    result = brute_force_method_elastic_collision(pos=pos,vel=vel)
    vel_next = result[1] + result[0]*dt #-------------------------------2
    pos_next = pos + vel_next*dt  #-------------------------------3
    return pos_next,vel_next

#kick-drift-kick, slower but more precised
@nb.njit()
def update_leapfrog(pos,vel):
    g = brute_force_method(pos=pos)
    vel_half = vel + 0.5*dt*g
    pos_next = pos + vel_half*dt
    g_next = brute_force_method(pos=pos_next)
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
    datasaver(init=True)
    counter = 0
    for time in np.linspace(0,t_max,time_steps):
        datasaver(t=time)
        # print("pos_M\n",pos_M)
        # print("vel_M\n",vel_M)
        #pos_M,vel_M = update_elastic_collision(pos=pos_M,vel=vel_M)
        pos_M,vel_M = update_leapfrog(pos=pos_M,vel=vel_M)
        if counter == 10:
            counter = 0
            E = total_K(vel_M) + total_U(pos_M)
            print("E: %d, E0: %d, Error: %0.3f %%" % (E,E0,100*(E-E0)/E0))
        print("Recording the data. Progress:%0.2f %%" % (100*time/t_max))
        counter += 1
        
if __name__ == '__main__':
    program_start = time.time()
    #cProfile.run('main()',sort="cumulative")
    main()
    program_end = time.time()
    print(f"N={N}, Total computation time: {program_end-program_start}s, filename: {filename}")