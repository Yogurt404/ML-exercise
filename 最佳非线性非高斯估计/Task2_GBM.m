% Task 2
% Grid Based Method

%% Real State
clc
clear all
load('x.mat');  % call saved x values
load('z.mat');  % call saved z values
Q = 10;
R = 1;

%% Grid Based Method
N = 10;
Center = zeros(N,1); % Initial grid center
G = linspace(-50,50,N)';
w = ones(N,1)/N; % Initial weights

for k= 1:100 % K time
    % Compute new grid center
    for i = 1:N
        Center(i) = G(i)./2+25*G(i)./(1+G(i).^2)+8*cos(1.2*k);
    end
    
    % Compute updated weight
    w1 = zeros(N,1);
    for i = 1:N
        for j = 1:N
            w1(i) = w1(i)+w(j)*(1./sqrt(2*Q*pi))*exp(-(G(i)-Center(j)).^2./(2./Q));
        end
    end
    
    % Compute new weight
    for i = 1:N
        w(i) = w1(i)*(1/sqrt(2*R*pi))*exp(-(G(i).^2./20-z(k)).^2./(2./R));
    end
    
    % Weight normalization
    w = w./sum(w);
    
    % The estimated value of x
    Grid_x(k) = G'*w;
    E_GBM(k) = (x(k) - Grid_x(k)).^2;
end
RMSE_GBM = sqrt(mean(E_GBM))

%% Data visualization
figure()
plot(x)
hold on
plot(Grid_x,'--r')
xlabel('Time')
ylabel('State Value')
legend('Real State','Grid Based Method')
title('Real and estimated values based on grid based method')

        

    


