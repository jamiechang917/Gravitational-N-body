import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import time

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000
plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\Jamie Chang\Desktop\ML_Snake_Game\FINAL\FFMPEG\bin\ffmpeg.exe'

# Updates Notes
# Remove class Particles, try numba

#===========Problems need to be fixed=============#
# 1. Particles.generate_particles() needs to be improved
# 2.

#==============Simulation Parameters==============#
N = 100
dt = 0.01
t_max = 5
softening_factor = 0.1
#==============Physical Parameters================#
mass = 1
avg_vel  =0
G = 10
#==============Environment Parameters=============#
#x_range = [-50,50] # range of initial position of particles in x axis
#y_range = [-50,50] # range of initial position of particles in y axis
r_range = 10 # the max range of the distance from origin to initial position of particles
box_length = 30
#==============Videos Parameters==================#
FPS = 30
VIDEO_PATH = r"C:\Users\Jamie Chang\Desktop\GalacticSim\Python\videos"

#==============Others Parameters==================#
t = 0
time_steps = int(t_max/dt)
#=================================================#


def main():
    def generate_particles(N,r_range,avg_vel): # Initialization
        random_theta = np.random.uniform(0,2*np.pi,(N,1))
        pos_M = np.random.uniform(0,r_range,(N,1)) * np.concatenate((np.cos(random_theta),np.sin(random_theta)),axis=1)
        vel_M = avg_vel*np.concatenate((np.cos(random_theta),np.sin(random_theta)),axis=1)
        return pos_M, vel_M

    pos_M, vel_M = generate_particles(N,r_range,avg_vel)
    @nb.njit
    def brute_force_method():
        g = np.zeros((N,2)) # 2 for 2D
        for i in range(N):
            for j in range(N):
                r = pos_M[i] - pos_M[j]
                g[i] += -G*mass*(((np.linalg.norm(r)**2) + (softening_factor**2))**-1.5)*r
        return g

    #@nb.njit if use here, the movie stuck but pos_M keeps updating
    def update(vel_M=vel_M,pos_M=pos_M):
        #global t
        vel_M += brute_force_method()*dt
        pos_M += vel_M*dt
        #t += dt

    def animation():
        # Matplotlib animation
        fig, ax = plt.subplots(figsize = (10, 10),dpi=100)
        scat = ax.scatter([],[],s=1)

        def fig_init():
            ax.set_xlim(-box_length,box_length)
            ax.set_ylim(-box_length,box_length)
            return scat,

        def fig_update(frame,pos_M=pos_M):
            update()
            scat.set_offsets(pos_M)
            print("Recording the animation. Progress:%0.2f %%" % (100*frame/t_max))
            return scat,
        ani = FuncAnimation(fig=fig,func=fig_update,frames=np.linspace(0,t_max,time_steps),init_func=fig_init,blit=True,repeat=False,save_count=sys.maxsize)

        ani.save(VIDEO_PATH+r"\N_Body.mp4",fps=FPS)
        #plt.show()
        print(".mp4 file saved.")

    animation()


if __name__ == '__main__':
    program_start = time.time()
    main()
    program_end = time.time()
    print(f"N={N}, Total computation time: {program_end-program_start}s")

