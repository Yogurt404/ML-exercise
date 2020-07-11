%Task 1
clc
clear all

ob = 0;
Q = 10;
R = 1;
for ob =1:100000 
    WGN = randn();
    x0 = WGN; % Initial statement
    v0 = sqrt(Q).*WGN; % Initial process noise
    n = sqrt(R).*randn(1,100); % Observation noise collection
    v = sqrt(Q).*randn(1,100); % Process noise collection
    
    x(1) = x0./2 + 25*x0./(1+x0.^2) + 8 * cos(1.2) + v0;% The real state at the K = 1 moment
    z(1) = x(1) ^ 2/20+ n(1);% Observation at time k = 1
    
    for k = 2:100 % Set recursion times
        x(k) = x(k-1)./2 + (25*x(k-1))./(1+x(k-1)^2)+8*cos(1.2*k)+v(k-1); % The real state at time k
        z(k) = x(k)^2/20 + n(k); % Observation of the real state at time k
    end
    x1(ob) = x(1); % Monte Carlo observation for x1
    x50(ob) = x(50);% Monte Carlo observation for x50
    x100(ob) = x(100);% Monte Carlo observation for x100
end

% Plotting a probability density function
[f1,xi1] = ksdensity(x1); % Plotting a probability density function of the sample x1
[f2,xi2] = ksdensity(x50);% Plotting a probability density function of the sample x50
[f3,xi3] = ksdensity(x100);% Plotting a probability density function of the sample x100

% Plotting the pdfs p(x1)
figure()
histogram(x1,60,'Normalization','probability','FaceColor','b','FaceAlpha',1)
hold on;
plot(xi1,f1,'-b','linewidth',2)
hold on;



% Plotting the pdfs p(x50)

histogram(x50,60,'Normalization','probability','FaceColor','r','FaceAlpha',1)
hold on;
plot(xi2,f2,'-r','linewidth',2)
hold on;



% Plotting the pdfs p(x100)

histogram(x100,60,'Normalization','probability','FaceColor','y','FaceAlpha',1)
hold on;
plot(xi3,f3,'-y','linewidth',2)
hold on;
legend('pdf(x1)','x1','pdf(x50)','x50','pdf(x100)','x100')
title('Estimate of the probability density function of x1, x50 and x100')

% Plotting real state
figure()
plot(x)
ylabel('Real state')
xlabel('Time')
title('Real state value x(k) of the Monte Carlo method')

% Plotting observation of real state
figure()
plot(z)
ylabel('Observation')
xlabel('Time')
title('Observation value of real state z(k) of the Monte Carlo method')

%...Saved Data Sets Will be Used For Filter Files...
save('x.mat','x');% Save x values 
save('z.mat','z');% Save z values 
    