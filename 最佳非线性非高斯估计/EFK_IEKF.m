% Task 2 & Task 4
% Extended Kalman Filter & Iterated Extended Kalman Filter
%% Real State
Q = 10;
R = 1;
load('x.mat');  % call saved x values
load('z.mat');  % call saved z values

%% Task 2: Extended Kalman Filter & Task 4: Iterated Extended Kalman Filter
% Initialization
x0 = 0;
P0 = 1000; % Initial error covariance matrix
F0 = 1/2 + (25-25*x0^2)/(1+x0^2)^2; % Initial partial derivative
N = 10 ;% Iterative number

% Compute the prior estimate of the error covariance matrix
% at time k = 1

Pri_P(1) = F0*P0*F0'+ Q;

% Compute the prior estimate of the predicted value 
% at time k = 1
Pri_x(1) = x0/2 + (25*x0)/(1+x0^2)+ 8*cos(1.2);


% Compute the Jacobian matrix at the predicted value
% at time k = 1
H(1) = Pri_x(1)/Q;

% Update equation
% at time k = 1
K(1) = Pri_P(1)*H(1)'*inv(H(1)*Pri_P(1)*H(1)'+1); % Compute Kalman gain

% Compute the posterioir estimate of the predicted value 
% at time k = 1
Post_x(1) = Pri_x(1)+K(1)*(z(1)-(Pri_x(1)^2/20));

% Compute the posterior estimate of the error covariance matrix
% at time k = 1
Post_P(1) = (eye-K(1)*H(1))*Pri_P(1)*(eye-K(1)*H(1))'+K(1)*K(1)';

for k = 2:100 % Set the number of iterations
    %% Task 2
    % Extended Kalman Filter
    % Set up the iteration process
    % Stage 1: prediction
    
    F = 1/2 +(25-25*Post_x(k-1)^2)/(1+Post_x(k-1)^2)^2;% Compute the Jacobian matrix of the estimated value at time k-1
    Pri_x(k) = Post_x(k-1)/2 + (25*Post_x(k-1))/(1+Post_x(k-1)^2)+8*cos(1.2*k); % Compute the prior estimate of the predicted value at time k
    Pri_P(k) = F*Post_P(k-1)*F'+ Q;    % Compute the prior estimate of the error covariance matrix at time k
    H(k) = Pri_x(k)/Q;   % Compute the Jacobian matrix at the predicted value at time k
    
    % Stage 2: correction
    
    K(k) = Pri_P(k)*H(k)'*inv(H(k)*Pri_P(k)*H(k)'+1);    % Compute Kalman gain at time k
    Post_x(k) = Pri_x(k)+K(k)*(z(k)-(Pri_x(k)^2/20)); % Compute the posterioir estimate of the predicted value  at time k
    Post_P(k) = (eye-K(k)*H(k))*Pri_P(k)*(eye-K(k)*H(k))'+K(k)*K(k)';    % Compute the posterior estimate of the error covariance matrix at time k
    
    %% Task 4
    % Iterated Extended Kalman Filter
    for i = 1:N
        % Initialization
        Post_xi(1,i) = Post_x(1); % Iterative initial posterior estimate
        Post_pi(1,i) = Post_P(1); % Iterative initial posterior covariance
        Hi(1,i) = Post_xi(1,i)/Q; % Iterative initial Jacobian matrix
        Ki(1,i) = Pri_P(1)*Hi(1,i)'*inv(Hi(1,i)*Pri_P(1)*Hi(1,i)'+1); % Iterative initial Kalman gain
        Post_xi(1,i+1) = Pri_x(1)+Ki(1,i)*(z(1)-(Post_xi(1,i).^2./20-Hi(1,i)*Pri_x(1)-Post_xi(1,i))); % Compute iterative posterior estimate
        Post_Pi(1,i+1) = (eye-Ki(1,i)*Hi(1,i))*Pri_P(1); % Compute iterative posterior covariance
        
        % Stage 1: prediction 
        % The same prediction equation as EKF
        Post_xi(k,i) = Post_x(k);
        Post_pi(k,i) = Post_P(k);
        Hi(k,i) = Post_xi(k,i)/Q;
        
        % Stage 2 : correction
        Ki(k,i) = Pri_P(k)*Hi(k,i)'*inv(Hi(k,i)*Pri_P(1)*Hi(k,i)'+1);
        Post_xi(k,i+1) = Pri_x(k)+Ki(k,i)*(z(k)-(Post_xi(k,i).^2./20-Hi(k,i)*Pri_x(k)-Post_xi(k,i)));
        Post_Pi(k,i+1) = (eye-Ki(k,i)*Hi(k,i))*Pri_P(k);
    
    end
    E_EKF(k) = (x(k) - Post_x(k)).^2;
    E_IEKF(k) = (x(k)-Post_xi(k,11)).^2;
end
RMSE_EKF = sqrt(mean(E_EKF))
RMSE_IEKF = sqrt(mean(E_IEKF))
%% Data Visualization
figure()
plot(x,'b')
hold on;
% plot(Post_x,'--r')
% hold on;
plot(Post_xi(:,11),'--r')
xlabel('Time')
ylabel('State Value')
% legend('Real State','Extended Kalman Filter')
% title('Real and estimated values based on extended Kalman filter')
% legend('Real State','Extended Kalman Filter','Iterated Extended Kalman Filter')
% title('Real and estimated values based on extended Kalman filter algorithm and iterated extended Kalman filter')
legend('True State','Iterated Extended Kalman Filter')
title('True and estimated values based on Iterated Extended Kalman filter')
