close all
clear all
clc
%===========Parameters===========%
% Global everything
global G N dt t_max t mass radius avg_velocity pos_M vel_M time_steps
% Simulation Parameters
N = 500; % numbers of particles
dt = 0.01;
t_max = 3;
t = 0;
time_steps = int32(t_max/dt);

% Properties of Particles
mass = 1;
radius = 1;
avg_velocity = 10;
E0 = N*0.5*mass*(avg_velocity^2);
x_range = 20; % range of initial position in x axis
y_range = 20; % range of initial position in y axis

% Properties of Squared Box
box_length = 200;

% Physical Constant
G = 6.67*10^1;
%====================================%

% Initialize position and velocity matrices
pos_M = [];
vel_M = [];
for i=1:N
    pos_M(end+1,:) = init_position_2D(-x_range,x_range,-y_range,y_range);
    vel_M(end+1,:) = init_velocity_2D(avg_velocity);
end

% Main
clear M
h = figure;
video = VideoWriter('nBody.avi');
open(video);
for timestep=1:time_steps
    scatter(pos_M(:,1),pos_M(:,2),'filled')
    E = 0;
    for i=1:N
        vel_M(i,:) = vel_M(i,:) + dt.*calculate_gravitational_acc(i);
        pos_M(i,:) = pos_M(i,:) + dt.*vel_M(i,:);
        E = E + 0.5*mass*(norm(vel_M(i,:))^2);
    end
    t = t+dt;
    
    axis equal;
    xlim([-box_length/2,box_length/2]);
    ylim([-box_length/2,box_length/2]);
    %M(timestep) = getframe(h);
    writeVideo(video,getframe(h));
    sprintf('Time: %0.3f, Progress: %0.1f %%, E0: %0.3f, E: %0.3f',t,(100*timestep/time_steps),E0,E)
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
   global N pos_M G mass
   acc = [0,0];
   for i=1:N
      r = pos_M(particle_index,:) - pos_M(i,:);
      acc = acc + (-G*mass/norm(r+10^-3)^3).*r;
   end
end

