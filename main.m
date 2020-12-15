clear all
%====================================%
% Global everything
global G N dt t_max t mass radius avg_velocity pos_M vel_M
% Simulation Parameters
N = 100; % numbers of particles
dt = 0.01;
t_max = 60;
t = 0;

% Properties of Particles
mass = 1;
radius = 1;
avg_velocity = 10;
x_range = 20; % range of initial position in x axis
y_range = 20; % range of initial position in y axis

% Properties of Squared Box
box_length = 100;

% Physical Constant
G = 6.67*10^1;
%====================================%

pos_M = [];
vel_M = [];

for i=1:N
    pos_M(end+1,:) = init_position_2D(-x_range,x_range,-y_range,y_range);
    vel_M(end+1,:) = init_velocity_2D(avg_velocity);
end


clear M
h = figure;
M(1)=getframe;
while t < t_max
    scatter(pos_M(:,1),pos_M(:,2),'filled')
    for i=1:N
        vel_M(i,:) = vel_M(i,:) + dt.*calculate_gravitational_acc(i);
        pos_M(i,:) = pos_M(i,:) + dt.*vel_M(i,:);
    end
t = t+dt;
axis equal;
xlim([-box_length/2,box_length/2])
ylim([-box_length/2,box_length/2])
M(end+1) = getframe(h);
clf
end
movie(gcf,M,2,24);

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
      if i ~= particle_index
          r = pos_M(particle_index,:) - pos_M(i,:);
          acc = acc + (-G*mass/norm(r)^3).*r;
      end
   end
end

