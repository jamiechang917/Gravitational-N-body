import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000
#mpl.use('svg') # Change the backend of mpl, it may cause blank video.
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import csv
import numpy as np
import time

plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\Jamie Chang\Desktop\ML_Snake_Game\FINAL\FFMPEG\bin\ffmpeg.exe'
#==============Videos Parameters==================#
box_length = 30
FPS = 30
VIDEO_PATH = r"C:\Users\Jamie Chang\Desktop\GalacticSim\Python\videos"
#=================================================#
CSV_PATH = r"C:\Users\Jamie Chang\Desktop\GalacticSim\Python\N_Body_Sun_Dec_27_002411_2020.csv"

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
    fig, ax = plt.subplots(figsize = (10, 10),dpi=100)
    scat = ax.scatter([],[],s=1)
    t_max = float(parameters[3])
    dt = float(parameters[2])
    N = int(parameters[1])
    timesteps = int(t_max/dt)

    def fig_init():
        ax.set_xlim(-box_length,box_length)
        ax.set_ylim(-box_length,box_length)
        return scat,

    def fig_update(timestep):
        # 2 represents dimensions
        t,pos_M,vel_M = np.float(data[3*timestep][0]),np.array(data[1+3*timestep]).astype(np.float).reshape((N,2)),np.array(data[2+3*timestep]).astype(np.float).reshape((N,2))
        scat.set_offsets(pos_M)
        print("Recording the animation. Progress:%0.2f %%" % (100*timestep/timesteps))
        return scat,

    ani = FuncAnimation(fig=fig,func=fig_update,frames=np.arange(timesteps),init_func=fig_init,blit=True,repeat=False,save_count=sys.maxsize)
    ani.save(VIDEO_PATH+r"\N_Body.mp4",fps=FPS)
    # plt.show()
    print(".mp4 file saved.")

if __name__ == '__main__':
    program_start = time.time()
    animation()
    program_end = time.time()
    print(f"Total computation time: {program_end-program_start}s")
