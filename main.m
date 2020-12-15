close all
clear all
clc
rng('default')
%===========Parameters===========%
% Global everything
global G N dt t_max t mass avg_velocity pos_M vel_M time_steps softening_factor
% Simulation Parameters
N = 100; % numbers of particles
dt = 0.01;
t_max = 3;
t = 0;
time_steps = int32(t_max/dt);
softening_factor = 1; % This factor should be small, cannot be zero otherwise will cause error

% Properties of Particles
mass = 1;
avg_velocity = 10;
x_range = 10; % range of initial position in x axis
y_range = 10; % range of initial position in y axis

% Properties of Squared Box
box_length = 100;

% Physical Constant
G = 1;
%====================================%

% Initialize position and velocity matrices
pos_M = [];
vel_M = [];
for i=1:N
    pos_M(end+1,:) = init_position_2D(-x_range,x_range,-y_range,y_range);
    vel_M(end+1,:) = init_velocity_2D(avg_velocity);
end

U0 = calculate_total_potential_energy();
K0 = N*0.5*mass*(avg_velocity^2);
E0 = K0 + U0;

% Main
clear M
h = figure;
video = VideoWriter('nBody.avi');
open(video);
for timestep=1:time_steps
    scatter(pos_M(:,1),pos_M(:,2),5,'filled')
    K = 0;
    for i=1:N
        vel_M(i,:) = vel_M(i,:) + dt.*calculate_gravitational_acc(i);
        pos_M(i,:) = pos_M(i,:) + dt.*vel_M(i,:);
        K = K + 0.5*mass*(norm(vel_M(i,:))^2);
    end
    U = calculate_total_potential_energy();
    E = K+U;
    t = t+dt;
    
    axis equal;
    xlim([-box_length/2,box_length/2]);
    ylim([-box_length/2,box_length/2]);
    %M(timestep) = getframe(h);
    writeVideo(video,getframe(h));
    sprintf('Time: %0.3f, Progress: %0.1f %%, E0: %d, E: %d',t,(100*timestep/time_steps),E0,E)
    clf
end
%movie(gcf,M,3,24);
close(video);



%=========Functions=========%
function pos = init_position_2D(xmin,xmax,ymin,ymax)
     pos = [xmin+(xmax-xmin).*rand(),ymin+(ymax-ymin).*rand()];
end

function vel = init_velocity_2D(init_velocity)
     theta = 2*pi*rand();
     vel = init_velocity.*[cos(theta), sin(theta)];
end

function acc = calculate_gravitational_acc(particle_index)
   global N pos_M G mass softening_factor
   acc = [0,0];
   for i=1:N
      r = pos_M(particle_index,:) - pos_M(i,:);
      acc = acc + (-G*mass/((norm(r)^2+softening_factor^2)^1.5)).*r; % add softening factor to avoid acc diverging.
   end
end

function potential = calculate_total_potential_energy()
    global pos_M G mass N
    potential = 0;
    for i=1:N-1
       for j=i+1:N
            r = norm(pos_M(i,:) - pos_M(j,:));
            potential = potential + (-G*(mass^2)/r);
       end
    end
end

