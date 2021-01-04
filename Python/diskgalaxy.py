import numpy as np
from scipy.integrate import quad,dblquad

#The Miyamoto model

#checked
def potential(G,M,a,b,R,z): # a=0 reduces to the plummer distribution (spherical), b=0 reduces to Kuzmin disk (thin)
    return -G*M*((R**2+(a+(z**2+b**2)**0.5)**2)**(-0.5))

#checked
def partial_potential(G,M,a,b,R,z,h=0.0001,target='R'):
    if target == 'R':
        #return (-potential(G=G,M=M,a=a,b=b,R=R+2*h,z=z)+8*potential(G=G,M=M,a=a,b=b,R=R+h,z=z)-8*potential(G=G,M=M,a=a,b=b,R=R-h,z=z)+potential(G=G,M=M,a=a,b=b,R=R-2*h,z=z))/(12*h)
        return G*M*R*((R**2+(a+(z**2+b**2)**0.5)**2)**(-1.5))
    elif target == 'z':
        #return (-potential(G=G,M=M,a=a,b=b,R=R,z=z+2*h)+8*potential(G=G,M=M,a=a,b=b,R=R,z=z+h)-8*potential(G=G,M=M,a=a,b=b,R=R,z=z-h)+potential(G=G,M=M,a=a,b=b,R=R,z=z-2*h))/(12*h)
        return G*M*z*(a+(z**2+b**2)**0.5)*((z**2+b**2)**-0.5)*(((a+(z**2+b**2)**0.5)**2+R**2)**-1.5)
    else:
        raise AttributeError("Target of partial potential is wrong.")

#checked
def double_partial_potential(G,M,a,b,R,z,h=0.0001,target='RR'):
    if target == 'RR':
        #return (-partial_potential(G=G,M=M,a=a,b=b,R=R+2*h,z=z)+8*partial_potential(G=G,M=M,a=a,b=b,R=R+h,z=z)-8*partial_potential(G=G,M=M,a=a,b=b,R=R-h,z=z)+partial_potential(G=G,M=M,a=a,b=b,R=R-2*h,z=z))/(12*h)
        return G*M*((R**2+(a+(z**2+b**2)**0.5)**2)-3*(R**2))*((R**2+(a+(z**2+b**2)**0.5)**2)**(-2.5))
    else:
        raise AttributeError("Target of partial potential is wrong.")

#checked
def mass_density(M,a,b,R,z): # three-dimension density
    return (b**2)*(M/(4*np.pi))*( (a*(R**2)+(a+3*((z**2+b**2)**0.5))*((a+((z**2+b**2)**0.5))**2)) / (((R**2+(a+((z**2+b**2)**0.5))**2)**2.5)*((z**2+b**2)**1.5)) )

def disk_density(M,a,b,R): # exponetial decay
    #softening_radius = 0.25*a
    #normalization_constant = M/(2*np.pi*a*(a+softening_radius)*np.exp(-softening_radius/a))
    #return normalization_constant*np.exp(-((R**2+2*(softening_radius**2))**0.5/a))
    #return np.exp(-((R**2+2*(softening_radius**2))**0.5/a))

    return a*M/(2*np.pi*((R**2+a**2)**1.5))
    # def mass_density(z,M,a,b,R):
    #     return (b**2)*(M/(4*np.pi))*( (a*(R**2)+(a+3*((z**2+b**2)**0.5))*((a+((z**2+b**2)**0.5))**2)) / (((R**2+(a+((z**2+b**2)**0.5))**2)**2.5)*((z**2+b**2)**1.5)) )
    # return quad(mass_density,-b,b,args=(M,a,b,R))[0]

#checked (z independent? )
def circular_velocity(G,M,a,b,R,z):
    return (R*partial_potential(G=G,M=M,a=a,b=b,R=R,z=0,target='R'))**0.5

#checked
def angular_velocity_squared(G,M,a,b,R):
    return G*M*((R**2+(a+(0**2+b**2)**0.5)**2)**(-1.5))

#checked
def partial_angular_velocity_squared(G,M,a,b,R,h=0.0001,target='R'):
    if target == 'R':
        #return (-angular_velocity_squared(G=G,M=M,a=a,b=b,R=R+2*h)+8*angular_velocity_squared(G=G,M=M,a=a,b=b,R=R+h)-8*angular_velocity_squared(G=G,M=M,a=a,b=b,R=R-h)+angular_velocity_squared(G=G,M=M,a=a,b=b,R=R-2*h))/(12*h)
        return -3*G*M*R*((R**2+(a+(0**2+b**2)**0.5)**2)**(-2.5))
    else:
        raise AttributeError("Target of partial potential is wrong.")

#checked
def epicyclic_frequency_squared(G,M,a,b,R):
    #return (R*partial_angular_velocity_squared(G=G,M=M,a=a,b=b,R=R)+4*angular_velocity_squared(G=G,M=M,a=a,b=b,R=R))  # it should yield same result of below
    return (double_partial_potential(G=G,M=M,a=a,b=b,R=R,z=0)+3*angular_velocity_squared(G=G,M=M,a=a,b=b,R=R))

#checked
def minimum_radial_velocity_dispersion(G,M,a,b,R):
    return 3.36*G*disk_density(M=M,a=a,b=b,R=R)/(epicyclic_frequency_squared(G=G,M=M,a=a,b=b,R=R)**0.5)

def radial_velocity_squared_mean(G,M,a,b,R,Q):
    softening_radius = 0.25*a # typically 0.25 scale length
    critical_radius = 2.4*a # typically 2~3 scale length
    def radial_velocity_squared_mean_without_normalization(G,M,a,b,R):
        return np.exp(-((R**2+2*(softening_radius**2))**0.5/a))
    radial_velocity_dispersion_squared_at_critical_radius = radial_velocity_squared_mean_without_normalization(G=G,M=M,a=a,b=b,R=critical_radius)
    minimum_radial_velocity_dispersion_squared_at_critical_radius = minimum_radial_velocity_dispersion(G=G,M=M,a=a,b=b,R=critical_radius)**2
    normalization_constant = (Q**2)*minimum_radial_velocity_dispersion_squared_at_critical_radius/radial_velocity_dispersion_squared_at_critical_radius

    return normalization_constant*radial_velocity_squared_mean_without_normalization(G=G,M=M,a=a,b=b,R=R)


def azimuthal_velocity_dispersion_square(G,M,a,b,R,Q):
    return radial_velocity_squared_mean(G=G,M=M,a=a,b=b,R=R,Q=Q)*(epicyclic_frequency_squared(G=G,M=M,a=a,b=b,R=R)/(4*angular_velocity_squared(G=G,M=M,a=a,b=b,R=R)))

def azimuthal_velocity_mean_squared(G,M,a,b,R,z,Q):
    part = radial_velocity_squared_mean(G=G,M=M,a=a,b=b,R=R,Q=Q)*(1-(epicyclic_frequency_squared(G=G,M=M,a=a,b=b,R=R)/(4*angular_velocity_squared(G=G,M=M,a=a,b=b,R=R)))-(2*R/a))
    Vc = circular_velocity(G=G,M=M,a=a,b=b,R=R,z=z)**2
    # if Vc+part < 0:
    #     raise ValueError("azimuthal_velocity_mean_squared is negative! (diskgalaxy.py)")
    return max(Vc+part,0)

def vertical_velocity_dispersion_square(G,M,a,b,R):
    return np.pi*G*disk_density(M=M,a=a,b=b,R=R)*b

# velocity of azimuthal is mean_vel with SD (azimuthal_dispersion)
# velocity of radial and vertical are obtained from zero with SD(dispersion)

def pdf(M,a,b,R_lower,R_upper,z_lower,z_upper): # probability function of density (R must be positive)
    def f(R,z,M=M,a=a,b=b):
        return mass_density(M=M,a=a,b=b,R=R,z=z)*2*np.pi*R
    z_lower,z_upper,R_lower,R_upper = abs(z_lower),abs(z_upper),abs(R_lower),abs(R_upper)
    if R_lower>R_upper:
        R_lower,R_upper = R_upper,R_lower
    if z_lower>z_upper:
        z_lower,z_upper = z_upper,z_lower
    if R_lower == R_upper:
        return 2*dblquad(f,z_lower,z_upper,lambda R:0,lambda R:R_upper)[0]/M
    if z_lower == z_upper:
        return 2*dblquad(f,0,z_upper,lambda R:R_lower,lambda R:R_upper)[0]/M
    return dblquad(f,z_lower,z_upper,lambda R:R_lower,lambda R:R_upper)[0]/M # if lower bound is negative, scipy.dblquad will return zero.


import matplotlib.pyplot as plt
def generate_particles(N,m,a,b,R_max,z_max,space_split,G,Q):
    r = np.linspace(0,R_max,space_split)
    z = np.linspace(-z_max,z_max,2*space_split)
    dr,dz = abs(r[1]-r[0]),abs(z[1]-z[0])
    pos = []
    vel = []
    N_predict = 0
    for i in range(space_split):
        for j in range(2*space_split):
            N_predict += int(pdf(m*N,a,b,r[i]-0.5*dr,r[i]+0.5*dr,z[j]-0.5*dz,z[j]+0.5*dz)*N)
    for i in range(space_split):
        for j in range(2*space_split):
            num = int(pdf(m*N,a,b,r[i]-0.5*dr,r[i]+0.5*dr,z[j]-0.5*dz,z[j]+0.5*dz)*N)
            for _ in range(num):
                R = r[i]+np.random.uniform(-0.5*dr,0.5*dr)
                Z = z[j]+np.random.uniform(-0.5*dz,0.5*dz)
                if R>0:
                    theta = np.random.uniform(0,2*np.pi)
                    p = np.array([R*np.cos(theta),R*np.sin(theta),Z])
                    sigma_R = radial_velocity_squared_mean(G=G,M=m*N_predict,a=a,b=b,R=R,Q=Q)**0.5
                    sigma_z = vertical_velocity_dispersion_square(G=G,M=m*N_predict,a=a,b=b,R=R)**0.5
                    sigma_azimuthal = azimuthal_velocity_dispersion_square(G=G,M=m*N_predict,a=a,b=b,R=R,Q=Q)**0.5
                    mu_azimuthal = azimuthal_velocity_mean_squared(G=G,M=m*N_predict,a=a,b=b,R=R,z=Z,Q=Q)**0.5
                    v_R = np.random.normal(0,sigma_R)
                    v_z = np.random.normal(0,sigma_z)
                    v_azimuthal = np.random.normal(mu_azimuthal,sigma_azimuthal)
                    v = np.array([-v_azimuthal*np.sin(theta)+v_R*np.cos(theta),v_azimuthal*np.cos(theta)+v_R*np.sin(theta),v_z])
                    pos.append(p)
                    vel.append(v)
    # pos = np.array(pos)
    # N = pos.shape[0]
    # for position in pos:
    #     Z = position[2]
    #     R = np.linalg.norm(position[0:2])
    #     theta = np.arccos(position[0]/R)
    #     if R*np.cos(theta) != position[0]:
    #         raise ValueError
    #     sigma_R = radial_velocity_squared_mean(G=G,M=m*N,a=a,b=b,R=R,Q=Q)**0.5
    #     sigma_z = vertical_velocity_dispersion_square(G=G,M=m*N,a=a,b=b,R=R)**0.5
    #     sigma_azimuthal = azimuthal_velocity_dispersion_square(G=G,M=m*N,a=a,b=b,R=R,Q=Q)**0.5
    #     mu_azimuthal = azimuthal_velocity_mean_squared(G=G,M=m*N,a=a,b=b,R=R,z=Z,Q=Q)**0.5
    #     v_R = np.random.normal(0,sigma_R)
    #     v_z = np.random.normal(0,sigma_z)
    #     v_azimuthal = np.random.normal(mu_azimuthal,sigma_azimuthal)
    #     v = np.array([-v_azimuthal*np.sin(theta)+v_R*np.cos(theta),v_azimuthal*np.cos(theta)+v_R*np.sin(theta),v_z])
    #     vel.append(v)

    # vel = np.zeros((pos.shape[0],3))
    # for i in range(pos.shape[0]):
    #     vel[i] = np.cross(np.array((0,0,1)),pos[i]/np.linalg.norm(pos[i]))*circular_velocity(G=G,M=N*m,a=a,b=b,R=np.linalg.norm(pos[i][0:2]),z=pos[i][2])
    pos = np.array(pos)
    vel = np.array(vel)
    return pos,vel

# N = 2000
# M_galaxy = np.float(1.5*(10**12))
# mass = (M_galaxy/N)
# mass = 1
# G = 1
# a = 1 # https://arxiv.org/pdf/astro-ph/9710197.pdf (a=2.1,b=0.21,R_max=40,z_max=3,space_split=30)
# b = 1
# R_max = 5
# z_max = 5
# space_split = 10
# R = 0.026
# z = -0.052
# M = N*mass
# Q = 1.2

# R = np.linspace(-30,30,100)
# z = np.linspace(-10,10,100)
# R,z = np.meshgrid(R,z)
# p = circular_velocity(G=G,M=M,a=a,b=b,R=R,z=z)
# fig = plt.figure(dpi=100,figsize=(15,5))
# ax = fig.add_subplot(111)
# # ax.plot(R,p)
# im = ax.contourf(R,z,p,cmap='hot',levels=np.linspace(p.min(),p.max(),1000))
# ax.set_title("The logarithm of mass density a=0, b=1")
# ax.set_xlabel("R (kpc)")
# ax.set_ylabel("z (kpc)")
# fig.colorbar(im)
# #plt.savefig("densityb1.png")
# plt.show()


# print("Vc",circular_velocity(G,M,a,b,R,z))
# print("angular",angular_velocity_squared(G,M,a,b,R))
# print("k",epicyclic_frequency_squared(G,M,a,b,R))
# print("sigma_r",radial_velocity_squared_mean(G,M,a,b,R,Q)**0.5)
# print("sigma_z",vertical_velocity_dispersion_square(G,M,a,b,R)**0.5)
# print("sigma_azi",azimuthal_velocity_dispersion_square(G,M,a,b,R,Q)**0.5)
# print("V_phi",azimuthal_velocity_mean_squared(G,M,a,b,R,z,Q))

# R = np.linspace(0,10*a,1000)
# # #y = radial_velocity_squared_mean(G,M,a,b,R,Q)/vertical_velocity_dispersion_square(G,M,a,b,R)
# y = azimuthal_velocity_mean_squared(G,M,a,b,R,z,Q)
# plt.plot(R,y)
# print("check",min(y))

# y1 = np.log10(radial_velocity_squared_mean(G,M,a,b,R,Q)**0.5)
# y2 = np.log10(vertical_velocity_dispersion_square(G,M,a,b,R)**0.5)
# y3 = np.log10(azimuthal_velocity_dispersion_square(G,M,a,b,R,Q)**0.5)
# plt.plot(R,y1)
# plt.plot(R,y2)
# plt.plot(R,y3)

# plt.xlabel('R')
# plt.show()

# pos,vel = generate_particles(N=N,m=mass,a=a,b=b,R_max=R_max,z_max=z_max,space_split=space_split,G=G,Q=Q)
# plt.scatter(pos[:,0],pos[:,2],s=1)
# print(pos.shape[0])
# plt.show()
