% Task 2 & Task 4
% Particle Filtering & Regularized Gaussian Partilce Filter

%% Real State
clc
clear all
load('x.mat');  % call saved x values
load('z.mat');  % call saved z values
% Intialization
Q = 10;
R = 1;
% Initial state
x0 = 0.1;
% The number of particles
N = 30; 
Pri_x = x0;
Pri_xr = x0;
% Dimension
n = length(x0);
%Unit sphere volume
cc = pi;
%Form the optimal choice of bandwidth
aa=(8*(1/cc)*(n+4)*(2*sqrt(pi))^n)^(1/(n+4));
hopt=aa*N^(-1/(n+4));
resample_percent = 0.5;
Nr = resample_percent*N;

% Use a Gaussian distribution to randomly generate initial particles
for i = 1:N
    Pri_x(i) = x0+sqrt(Q)*randn(); % Partilcle Filter
    Pri_xr(i) = Pri_x(i); % Regularized Particle Filter
end

% Iteration process
for k = 1:100

%% Task 2: Rregularized Particle Filter
for i = 1:N
% Stage 1: Prediction
Post_x(i) = 1/2*Pri_x(i) + 25*Pri_x(i)/(1+Pri_x(i)^2)+8*cos(1.2*k)+sqrt(Q)*randn();
Post_z = Post_x(i)^2/20;
        
% Stage 2: Update
% At time k, update the weights based on the new measurement
q(i) = (1/sqrt(R)/sqrt(2*pi))*exp(-(z(k)-Post_z).^2/(2*R));
        
end
    
% Normalise the weights
q = q./sum(q);
    
% Stage 3: Resampling
for i = 1:N
         Pri_x(i) = Post_x(find(rand <= cumsum(q),1));
end

% State estimation
est_x(k)= mean(Pri_x);
    
%% Task 4: Regularized Particle Filter
for i = 1:N
% Stage 1: prediction

Post_xr(i) = 0.5*Pri_xr(i) + 25*Pri_xr(i)/(1+Pri_xr(i)^2)+8*cos(1.2*(k))+sqrt(Q)*randn();
Post_zr = Post_xr(i)^2/20;
        
% Stage 2: update

% At time k, update the weights based on the new measurement
qr(i) = (1/sqrt(R)/sqrt(2*pi*R))*exp(-(z(k)-Post_zr).^2/(2*R));       
end
    

%Stage 3: resampling
    
    % Normalise the weights
    qr = qr./sum(qr);
    
    % Compute empirical covariance of particles
    emp_cov=cov(Post_xr');
    % Form D'*D=emp_cov
    dd=chol(emp_cov)';

 
    % Posterior particle selection
    xr(1) = min(Post_xr)-std(Post_xr);
    xr(Nr) = max(Post_xr) + std(Post_xr);
    F1 = (xr(Nr)-xr(1))/(Nr-1);
    
     for i = 2:Nr-1
       xr(i) = xr(i-1)+F1;
     end

    
    % Optimal kernel density function

    for i = 1:Nr
        qr1(i) = 0;
        for j = 1:N
            Norm_xr = norm(inv(dd) * (xr(i)-Post_xr(j)));
            if Norm_xr<hopt
                qr1(i) = qr1(i)+ qr(j)*(n+2)*(1-Norm_xr^2/hopt^2)/2/cc/hopt^2/det(dd);

            end
        end
    end
    
    % Weight normalization
    qr1=qr1./sum(qr1);
    

% Resample for the regularized particle filter. (Page 474-Rule 6)
    for i = 1 : N
        u = rand();           % uniform random number between 0 and 1
        qtempsum = 0;
        for j = 1 : Nr
            qtempsum = qtempsum + qr1(j);
            if qtempsum >= u
                Pri_xr(i) = xr(j);
                break;
            end
        end
    end

    
    est_xr(k) = mean(Pri_xr);
    



end
E_PF= (x - est_x).^2;
E_RPF = (x - est_xr).^2;
RMSE_PF = sqrt(mean(E_PF))
RMSE_RPF = sqrt(mean(E_RPF))

%% Data visualization
figure()
plot(x,'b','linewidth',2)
hold on
%plot(est_x,'--r')
%hold on
plot(est_xr,'--r','linewidth',1.5)
xlabel('Time')
ylabel('State Value')
% legend('Real State','Particle Filter','Regularized Particle Filter')
% title('Real and estimated values based on Particle Filter')
legend('True State','Particle Filter')
title('True and estimated values based on Rgularized Particle Filter')












