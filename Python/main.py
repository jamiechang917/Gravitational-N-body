import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import time

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000
plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\Jamie Chang\Desktop\ML_Snake_Game\FINAL\FFMPEG\bin\ffmpeg.exe'

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

class Particles():
    def __init__(self,number_of_particles,mass,avg_vel,r_range):
        self.N = number_of_particles
        self.m = mass
        self.init_vel = avg_vel
        #self.x_range = x_range
        #self.y_range = y_range
        self.r_range = r_range
        self.avg_vel = avg_vel

        self.pos_M, self.vel_M = self.generate_particles(r_range=self.r_range,avg_vel=self.avg_vel)

    def generate_particles(self,r_range,avg_vel): # Initialization
        random_theta = np.random.uniform(0,2*np.pi,(N,1))
        pos_M = np.random.uniform(0,r_range,(N,1)) * np.concatenate((np.cos(random_theta),np.sin(random_theta)),axis=1)
        vel_M = avg_vel*np.concatenate((np.cos(random_theta),np.sin(random_theta)),axis=1)
        return (pos_M, vel_M)

def main():
    P = Particles(number_of_particles=N,mass=mass,avg_vel=avg_vel,r_range=r_range)

    def brute_force_method(P:Particles):
        g = np.zeros((N,2)) # 2 for 2D
        for i in range(N):
            for j in range(N):
                r = P.pos_M[i] - P.pos_M[j]
                g[i] += -G*mass*(((np.linalg.norm(r)**2) + (softening_factor**2))**-1.5)*r
        return g


    def update(P:Particles):
        global t
        P.vel_M += brute_force_method(P=P)*dt
        P.pos_M += P.vel_M*dt
        t += dt

    def animation():
        # Matplotlib animation
        fig, ax = plt.subplots(figsize = (10, 10),dpi=100)
        scat = ax.scatter([],[],s=1)

        def fig_init():
            ax.set_xlim(-box_length,box_length)
            ax.set_ylim(-box_length,box_length)
            return scat,

        def fig_update(frame):
            update(P=P)
            scat.set_offsets(P.pos_M)
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

