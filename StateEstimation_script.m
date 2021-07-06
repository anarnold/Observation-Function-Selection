% Script file to run state estimation simulations with known parameters
%
% Authors: Leah Mitchell, Andrea Arnold <anarnold@wpi.edu>
% Last updated: July 5, 2021
%
% Corresponds with the manuscript:
% 
% L. Mitchell and A. Arnold (2021) Analyzing the Effects of Observation 
% Function Selection in Ensemble Kalman Filtering for Epidemic Models,
% Accepted for publication in Mathematical Biosciences. 
% DOI: https://doi.org/10.1016/j.mbs.2021.108655

clear; close all; clc

%Load data
load data_set.mat

%Select case type ('case 1' through 'case 4')
casetype = 'case 4';

%Set ensemble size
N = 100; 

%Set SIR model parameters
Np = 9e4; 
lambda = 100; 
m = 0.02; 
b0 = 1800; 
b1 = 0.08; 

%Time points for observations
deltaT = 1/12;
tdata = 0:deltaT:10; 

%Generate prior ensembles
true_init = x0;

aS = 0.9*true_init(1); 
bS = 1.1*true_init(1); 

uniS = aS+(bS-aS).*rand(1,N);

aI = 0.9*true_init(2); 
bI = 1.1*true_init(2); 

uniI = aI+(bI-aI).*rand(1,N);

%Store ensembles
S = zeros(2,N); 
S(1,:) = uniS; 
S(2,:) = uniI; 

S = [S; zeros(1,N)];

%Ensemble statistics
xbar1 = (1/N)*sum(S,2);
gamma1 = ((S-xbar1)*transpose(S-xbar1))/(N-1);

%Store ensemble mean, +/- 2 standard deviation curves
fmean = zeros(length(data),3); 
fplus = zeros(length(data),3); 
fminus = zeros(length(data),3); 

fmean(1,:) = xbar1;
fplus(1,:) = transpose(abs(2*sqrt(diag(gamma1))))+fmean(1,:);
fminus(1,:) = transpose(-abs(2*sqrt(diag(gamma1))))+fmean(1,:);

stdC = 0.2;  
stdD = 1; 
D = stdD^2;     
nu = zeros(1,N);

options = odeset('RelTol',1e-8,'AbsTol',1e-8); 

%Main time loop
for j = 2:(length(data))

    %Prediction step
    for n = 1:N
        SIR_rhs =  @(t,Xt) [m.*(Np-Xt(1))-((b0.*(1+b1.*cos(2*pi*t)).*Xt(1).*Xt(2))./Np);
            ((b0.*(1+b1.*cos(2*pi*t)).*Xt(1).*Xt(2))./Np)-(lambda + m).*Xt(2);
            (b0.*(1+b1.*cos(2*pi*t)).*Xt(1).*Xt(2))./Np];
        
        ts = [tdata(j-1),tdata(j)];  
        
        x0 = [S(1,n);S(2,n);0];
        
        [~,Y] = ode15s(SIR_rhs,ts,x0,options);
        
        S(:,n) = transpose(Y(end,:))+stdC*randn(3,1);
        nu(n) = Y(end,3);
    end
    
    %Ensemble mean
    xbar = (1/N)*sum(S,2);

    %Analysis Step
    %Compute model prediction based on observation function selection
    if strcmp(casetype,'case 1')== 1
        yhat = IofT(S(1:2,:),1);
    elseif strcmp(casetype,'case 2')== 1
        yhat = IofT(S(1:2,:),0.7);
    elseif strcmp(casetype, 'case 3')== 1
        yhat = Integral(nu,1);
    elseif strcmp(casetype, 'case 4')== 1
        yhat = Integral(nu,0.7);
    else
        error('Case not found, try again');
    end
    
    %Kalman Gain
    yhatbar = (1/N)*sum(yhat,2);
    cross = ((S-xbar)*transpose(yhat-yhatbar))/(N-1);
    forecast = ((yhat-yhatbar)*transpose(yhat-yhatbar))/(N-1);
    K = cross/(forecast + D);
    ydata = data(j)+stdD*randn(1,N);
    
    %Update ensemble
    S = S+ K*(ydata-yhat);
    
    %Ensemble statistics
    fmean(j,:) = (1/N)*sum(S,2);
    gamma = ((S-transpose(fmean(j,:)))*transpose(S-transpose(fmean(j,:))))/(N-1);
    
    fplus(j,:) = transpose(abs(2*sqrt(diag(gamma))))+fmean(j,:);
    fminus(j,:) = transpose(-abs(2*sqrt(diag(gamma))))+fmean(j,:);
end

time = tdata;

%Plot results
GraphS=figure;
hold on
plot(time,fmean(:,1),'r-','LineWidth',2);
plot(time,fplus(:,1),'r--','LineWidth',1);
plot(time,fminus(:,1),'r--','LineWidth',1);
hold off
title('Susceptible Population');
xlabel('Time [years]');
ylabel('Population [individuals]');
set(gca,'FontSize',25);
xlim([0,10])

GraphI=figure;
hold on
plot(time,fmean(:,2),'r-','LineWidth',2);
plot(time,fplus(:,2),'r--','LineWidth',1);
plot(time,fminus(:,2),'r--','LineWidth',1);
if ( strcmp(casetype,'case 1')== 1 || strcmp(casetype,'case 2')== 1 )
    p=plot(time,data,'k.');
    p.MarkerSize = 18;
end
hold off
title('Infectious Population');
xlabel('Time [years]');
ylabel('Population [individuals]');
set(gca,'FontSize',25);
xlim([0,10])

if ( strcmp(casetype,'case 3')== 1 || strcmp(casetype,'case 4')== 1 )
    GraphNu=figure;
    hold on
    plot(time,fmean(:,3),'r-','LineWidth',2);
    plot(time,fplus(:,3),'r--','LineWidth',1);
    plot(time,fminus(:,3),'r--','LineWidth',1);
    p=plot(time,data,'k.');
    p.MarkerSize = 18;
    hold off
    title('Monthly Number of Cases');
    xlabel('Time [years]');
    ylabel('Population [individuals]');
    set(gca,'FontSize',25);
    xlim([0,10])
end

% Functions to compute observation model predictions
function I = IofT(S,rho)
    G = rho*[0 1];
    I = G*S;
end

function I = Integral(nu,rho)
    I = rho*nu;
end

