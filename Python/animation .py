import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000
#mpl.use('svg') # Change the backend of mpl, it may cause blank video.
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import sys
import csv
import numpy as np
import time

plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\Jamie Chang\Desktop\ML_Snake_Game\FINAL\FFMPEG\bin\ffmpeg.exe'
#==============Videos Parameters==================#
box_length = 20
time_scale = 1.9927 #Myrs
FPS = 24
VIDEO_PATH = r"C:\Users\Jamie Chang\Desktop\GalacticSim\Python\videos"
#==============Files Parameters===================#
CSV_PATH = r"C:\Users\Jamie Chang\Desktop\GalacticSim\Python\N_Body_Mon_Jan__4_205441_2021.csv"
COLLISION_MODE = False
# read csv file
def datareader(return_parameters=False):
    with open(CSV_PATH,'r',newline='') as csvfile:
        rows = list(csv.reader(csvfile,delimiter=','))
        if return_parameters == True:
            return rows[0]
        else:
            return rows[1:]

data = datareader()
parameters = datareader(return_parameters=True)
if COLLISION_MODE == True:
    N,G1_N,G2_N,dt,t_max,softening_factor,mass,G,a,G2_a,b,G2_b,Q,G2_Q,M_galaxy,G2_M,R_max,G2_R_max,z_max,G2_z_max,space_split,G2_space_split,G2_displacement,G2_speed = parameters[1:]
    G1_N = int(G1_N)
    G2_N = int(G2_N)
    N = int(N)
    t_max = float(t_max)
    dt = float(dt)
    timesteps = int(t_max/dt)
elif COLLISION_MODE == False:
    N,dt,t_max,softening_factor,mass,G,a,b,Q,M_galaxy,R_max,z_max,space_split = parameters[1:]
    N = int(N)
    t_max = float(t_max)
    dt = float(dt)
    timesteps = int(t_max/dt)

# sequence for energy scatter plot
t_list = []
K_list = []
U_list = []
E_list = []

############################################################3
# for timestep in range(timesteps):
#     if timestep % 50 == 0:
#         pos_M = np.array(data[1+3*timestep]).astype(np.float).reshape((N,3))
#         x = pos_M[:,0]
#         y = pos_M[:,1]
#         fig = plt.figure(dpi=200,figsize=(5,5))
#         plt.scatter(x,y,s=0.1,color='black')
#         plt.title(f"Q = {Q}, t = {timestep*dt}")
#         plt.xlim((-box_length,box_length))
#         plt.ylim((-box_length,box_length))
#         plt.xlabel("x (kpc)")
#         plt.ylabel("y (kpc)")
#         plt.savefig(f"CASE1_N{N}_e{softening_factor}_Q{Q}_t{timestep*dt}.png")

# animation
def animation():
    fig = plt.figure(figsize = (10, 10),dpi=200)
    ax_xy = fig.add_subplot(221,title="x-y plane",xlabel="x",ylabel="y")
    ax_information = fig.add_subplot(222)
    ax_xz = fig.add_subplot(223,title="x-z plane",xlabel="x",ylabel="z")
    ax_yz = fig.add_subplot(224,title="y-z plane",xlabel="y",ylabel="z")
    if COLLISION_MODE == True:
        scat_xy_G2 = ax_xy.scatter([],[],s=0.5,c='orange')
        scat_xz_G2 = ax_xz.scatter([],[],s=0.5,c='orange')
        scat_yz_G2 = ax_yz.scatter([],[],s=0.5,c='orange')
    scat_xy = ax_xy.scatter([],[],s=0.5)
    scat_xz = ax_xz.scatter([],[],s=0.5)
    scat_yz = ax_yz.scatter([],[],s=0.5)
    scat_E = ax_information.scatter([],[],c='black')
    scat_K = ax_information.scatter([],[],c='red')
    scat_U = ax_information.scatter([],[],c='blue')
    ax_information.legend((scat_E,scat_K,scat_U),('E','K','U'),loc='upper right')



    def fig_init():
        for ax in (ax_xy,ax_xz,ax_yz):
            ax.set_xlim(-box_length,box_length)
            ax.set_ylim(-box_length,box_length)
        fig.suptitle("Galaxy Simulation")
        return scat_xy,scat_xz,scat_yz

    def fig_update(timestep):
        information_row = data[3*timestep]
        if len(information_row) != 1: # if this row contains energy
            t = np.float(information_row[0])
            K = np.float(information_row[1])
            U = np.float(information_row[2]) #[1:-1]
            E = np.float(information_row[3])
            t_list.append(t)
            K_list.append(K)
            U_list.append(U)
            E_list.append(E)
            scat_K.set_offsets(np.array(list(zip(t_list,K_list))))
            scat_U.set_offsets(np.array(list(zip(t_list,U_list))))
            scat_E.set_offsets(np.array(list(zip(t_list,E_list))))
            fig.suptitle("Galaxy Simulation, t=%0.2fMyr" % (t*time_scale))
        else:
            t = np.float(information_row[0])
            fig.suptitle("Galaxy Simulation, t=%0.2fMyr" % (t*time_scale))
        if COLLISION_MODE == True:
            pos_M,vel_M = np.array(data[1+3*timestep]).astype(np.float).reshape((N,3)),np.array(data[2+3*timestep]).astype(np.float).reshape((N,3))
            pos_M_G1, pos_M_G2 = pos_M[0:G1_N], pos_M[G1_N::]
            scat_xy.set_offsets(pos_M_G1[:,[0,1]])
            scat_xz.set_offsets(pos_M_G1[:,[0,2]])
            scat_yz.set_offsets(pos_M_G1[:,[1,2]])
            scat_xy_G2.set_offsets(pos_M_G2[:,[0,1]])
            scat_xz_G2.set_offsets(pos_M_G2[:,[0,2]])
            scat_yz_G2.set_offsets(pos_M_G2[:,[1,2]])
        elif COLLISION_MODE == False:
            pos_M,vel_M = np.array(data[1+3*timestep]).astype(np.float).reshape((N,3)),np.array(data[2+3*timestep]).astype(np.float).reshape((N,3))
            scat_xy.set_offsets(pos_M[:,[0,1]])
            scat_xz.set_offsets(pos_M[:,[0,2]])
            scat_yz.set_offsets(pos_M[:,[1,2]])
        print("Recording the animation. Progress:%0.2f %%" % (100*timestep/timesteps))
        if COLLISION_MODE == True:
            return scat_xy,scat_xz,scat_yz,scat_xy_G2,scat_xz_G2,scat_yz_G2
        elif COLLISION_MODE == False:
            return scat_xy,scat_xz,scat_yz

    ani = FuncAnimation(fig=fig,func=fig_update,frames=np.arange(timesteps),init_func=fig_init,blit=True,repeat=False,save_count=sys.maxsize)
    ani.save(VIDEO_PATH+r"\N_Body_3D.mp4",fps=FPS)
    print(".mp4 file saved.")

if __name__ == '__main__':
    program_start = time.time()
    # animation()
    program_end = time.time()
    print(f"Total computation time: {program_end-program_start}s")