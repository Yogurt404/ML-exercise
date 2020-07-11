%% The cost of the last week
function [cost, stock2] = last_week(y, r, stock)

% Create a random demand
D = dw(1);
if stock <= r
    stock = stock +y;
else
    stock = stock;
end

% Update stock
stock2 = stock - D;

% Check if need pay penalty cost
if stock2 < 0
    stock2 = 0;
    penalty_flag = 1; 
else
    penalty_flag = 0;
end


% Compute return cost and short of stock penalty

% Compute return cost
if stock2 >= 0
    cost = stock2 * 10;

% Compute short of stock penalty
elseif penalty_flag == 1
    cost = 20;
end


end
