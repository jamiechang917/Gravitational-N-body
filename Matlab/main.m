clear all
clc;
%====================================%
% Global everything
global G N dt t_max t mass radius avg_velocity pos_M vel_M softening_factor dimension
% Simulation Parameters
dimension = 3; 
N1=100;
N2=100;
N = N1+N2; % numbers of particles
dt = 0.01;
t_max = 5;
t = 0;
softening_factor=0.1;
% Properties of Particles
mass = 1;
radius = 1;
avg_velocity = 5;
x_range = 10; % range of initial position in x axis
y_range = 10; % range of initial position in y axis
z_range = 10;

% Properties of Squared Box
box_length = 100;

% Physical Constant
G = 1;
%====================================%

pos_M = [];
vel_M = [];



clear M
csv=readmatrix('result.csv','NumHeaderLines',1);
if(size(csv,2)==0)
    A=["N","dt","t_max","softening_factor","mass","avg_velocity",G,"radius"];
    B=[N,dt,t_max,softening_factor,mass,avg_velocity,G,radius];
    writematrix(A,'result.csv');
    writematrix(B,'result.csv','Writemode','append');
    clear A;
    for i=1:N
        pos_M(end+1,:) = init_position_2D([-x_range,x_range,-y_range,y_range,-z_range,z_range]);
        vel_M(end+1,:) = init_velocity_2D(avg_velocity);
    end
else
    const=num2cell(csv(1,:))
    [N,dt,t_max,softening_factor,mass,avg_velocity,G,radius]=const{:};
    t=csv(end-2,1);
    p=csv(end-1,1:N*dimension);
    pos_M=reshape(p,dimension,[]);
    v=csv(end-1,1:N*dimension);
    vel_M=reshape(p,dimension,[]);
end
clear csv;


%======Main Program=====%
M=getframe();
counter=0;
E0=total_energy()
while t <= t_max
    datasaver();
    update_leapfrag();
    if counter==10
        counter=0;
        E=total_energy();
        fprintf("E:%d, E0:%d, Error:%.3f\n",E,E0,100*(abs(E-E0)/E0));
    end
    fprintf("Recording data. Progress:%.2f\n",100*(t/t_max));
    counter=counter+1;
    t=t+dt;
    scatter3(pos_M(1:N1,1),pos_M(1:N1,2),pos_M(1:N1,3),'b','filled');
    hold on;
    scatter3(pos_M(N1+1:N,1),pos_M(N1+1:N,2),pos_M(N1+1:N,3),'r','filled');
    axis(2*[-x_range,x_range,-y_range,y_range,-z_range,z_range]);
    M(end+1)=getframe();
    clf;
end
v = VideoWriter('./animation.avi','Indexed AVI');
open(v);
writeVideo(v,M(2:end));
fclose('all');


function pos = init_position_2D(range)
    global dimension 
    if dimension == 2
        pos = [range(1)+(range(2)-range(1)).*rand(),range(3)+(range(4)-range(3)).*rand()];
    else dimension == 3
        pos = [range(1)+(range(2)-range(1)).*rand(),range(3)+(range(4)-range(3)).*rand(),range(5)+(range(6)-range(5)).*rand()];
    end
end

function vel = init_velocity_2D(init_velocity)
    global dimension
    if dimension == 2
     theta = 2*pi*rand();
     vel = init_velocity.*[cos(theta), sin(theta)];
    elseif dimension == 3
        theta = 2*pi*rand();
        phi =  2*pi*rand();
        vel = init_velocity.*[sin(theta)*cos(phi), sin(theta)*sin(phi),sin(theta)];
    end
end

function E=total_energy()
    global vel_M pos_M mass G N mass
    K=0;
    for i=1:N
        v_norm=norm(vel_M(i,:));
        K=K+0.5*mass*v_norm^2;
    end
    U=0;
    for i=1:N
        for j=i+1:N
            r=norm(pos_M(i,:)-pos_M(j,:));
            U=U-G*mass^2/r;
        end
    end
    E=U+K
end

function update_leapfrag()
    global pos_M vel_M dt
    a = brute_force_method(pos_M);
    vel_half=vel_M+0.5*a*dt;
    pos_next=pos_M+vel_half*dt;
    a_next=brute_force_method(pos_next);
    vel_next=vel_half+0.5*dt*a_next;
    pos_M=pos_next;
    vel_M=vel_next;
end

function acc = brute_force_method(pos_M)
   global N  G mass softening_factor dimension
   acc=zeros(N,dimension);
   for i=1:N-1;
       for j=i+1:N
           r=pos_M(i,:)-pos_M(j,:);
           if norm(r)<=softening_factor
               acc(i,:)=acc(i,:)-G*mass*((norm(r)^2+softening_factor^2)^(-1.5))*r;
               acc(j,:)=acc(j,:)+G*mass*((norm(r)^2+softening_factor^2)^(-1.5))*r;
           else
               acc(i,:)=acc(i,:)-G*mass*r/(norm(r))^3;
               acc(j,:)=acc(j,:)+G*mass*r/(norm(r))^3;
           end
       end
   end
end

function elastic_collidision()
    global N pos_M vel_M radius
    pos_M=sortrows(pos_M);
    for i=1:N-1
        for j=i+1:N
            if pos_M(j,1)-pos_M(i,1)>2*radius
                continue;
            end
            r=pos_M(i,:)-pos_M(j,:);
            v=vel_M(i,:)-vel_M(j,:);
            if norm(r)<=2*radius && dot(v,r)>0
                vi=dot(vel_M(i,:),r)*r/norm(r)^2;
                vj=dot(vel_M(j,:),r)*r/norm(r)^2;
                vel_M(i,:)=vel_M(i,:)-vi+vj;
                vel_M(j,:)=vel_M(j,:)-vj+vi;
            end
        end
    end
end

function datasaver()
    global pos_M vel_M t
    writematrix(t,'result.csv','Writemode','append');
    p=reshape(pos_M,1,[]);
    writematrix(p,'result.csv','Writemode','append');
    v=reshape(vel_M,1,[]);
    writematrix(v,'result.csv','Writemode','append');
end