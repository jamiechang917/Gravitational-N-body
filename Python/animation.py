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
box_length = 30
FPS = 30
VIDEO_PATH = r"C:\Users\Jamie Chang\Desktop\GalacticSim\Python\videos"
#==============Files Parameters===================#
CSV_PATH = r"C:\Users\Jamie Chang\Desktop\GalacticSim\Python\N_Body_Mon_Dec_28_231611_2020.csv"

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

# animation
def animation():
    fig = plt.figure(figsize = (10, 10),dpi=100)
    ax_xy = fig.add_subplot(221,title="x-y plane",xlabel="x",ylabel="y")
    ax_information = fig.add_subplot(222)
    ax_xz = fig.add_subplot(223,title="x-z plane",xlabel="x",ylabel="z")
    ax_yz = fig.add_subplot(224,title="y-z plane",xlabel="y",ylabel="z")
    scat_xy = ax_xy.scatter([],[],s=0.5)
    scat_xz = ax_xz.scatter([],[],s=0.5)
    scat_yz = ax_yz.scatter([],[],s=0.5)

    t_max = float(parameters[3])
    dt = float(parameters[2])
    N = int(parameters[1])
    timesteps = int(t_max/dt)

    def fig_init():
        for ax in (ax_xy,ax_xz,ax_yz):
            ax.set_xlim(-box_length,box_length)
            ax.set_ylim(-box_length,box_length)
        return scat_xy,scat_xz,scat_yz

    def fig_update(timestep):
        t,pos_M,vel_M = np.float(data[3*timestep][0]),np.array(data[1+3*timestep]).astype(np.float).reshape((N,3)),np.array(data[2+3*timestep]).astype(np.float).reshape((N,3))
        scat_xy.set_offsets(pos_M[:,[0,1]])
        scat_xz.set_offsets(pos_M[:,[0,2]])
        scat_yz.set_offsets(pos_M[:,[1,2]])
        print("Recording the animation. Progress:%0.2f %%" % (100*timestep/timesteps))
        return scat_xy,scat_xz,scat_yz

    ani = FuncAnimation(fig=fig,func=fig_update,frames=np.arange(timesteps),init_func=fig_init,blit=True,repeat=False,save_count=sys.maxsize)
    ani.save(VIDEO_PATH+r"\N_Body_3D.mp4",fps=FPS)
    print(".mp4 file saved.")

if __name__ == '__main__':
    program_start = time.time()
    animation()
    program_end = time.time()
    print(f"Total computation time: {program_end-program_start}s")