clc
clear all

% Number of simulations
estimate_num = 200; 
estimate_num1 = 100;

% The limitation fo horzion
h_pred = 4; % Prediction horizon
h_ctrl = 3; % Control horzion


% Optimal fixed lever
r = 1;
% Total weeks
a =52;
% Range of order number
min_y = 0;
max_y = 6;
y = [min_y: max_y];


%% Task 2
for i = 1:estimate_num1
    display(i); % Observe the number of iterations
    [opt_y,cost_pred(i)] = cost(a,y,r,h_ctrl,h_pred,estimate_num);  
end
% Compute mean and variance
mean_cost = mean(cost_pred);
var_cost = std(cost_pred);
display(mean_cost);
display(var_cost);
%% Data visualization
figure()
histogram(cost_pred,10)
xlabel('costs of 52 weeks')
ylabel('number of instance')
title('The distribution of the total cost')
%mean(cost_pred)


%% Task 3

% Observe the total cost of the combination of different forecast and
% control horzion


for h_ctrl = 1:6
    display(h_ctrl)
    for h_pred = h_ctrl:6
        [opt_y1(:,h_ctrl,h_pred),cost_pred1(h_ctrl,h_pred)] = cost(a,y,h_ctrl,h_pred,estimate_num,r);
    end
end


figure()
bar3(cost_pred1)
ylabel('cotrol horizon')
xlabel('prediction horizon')
zlabel('total cost')

















