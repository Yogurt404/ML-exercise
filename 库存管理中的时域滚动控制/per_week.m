%% The cost of every week
function [stock2, cost] = per_week(y, r, stock)

% Create a random demand
D = dw(1);
if stock <= r
    stock = stock +y;
else
    stock = stock;
end

% Update stock
stock2 = stock - D;

% Check if need pay penalty
if stock2 < 0
    stock2 = 0;
    penalty_flag = 1; 
else
    penalty_flag = 0;
end

% Compute penalty cost and warehouse cost

% Compute short of stock penalty
if penalty_flag == 1
    cost = 20;
% Compute warehouse cost
elseif stock2 >= 0
    cost = stock2 * 5;
end

end