%% Compute total cost in every simulation
function [opt_y, cost_pred] = cost(a, y, r, h_ctrl, h_pred, estimate_num)

% Gain optimal y and check if need re-order
for stock = 0: 6
	[opt_y(stock + 1), order_flag(stock + 1)] = gain_opt_y(y, r, h_ctrl, h_pred, estimate_num, stock);
end


for j = 1: estimate_num
    % The initial stock
    stock = 0;
    % Compute total cost
    for i = 1: a - 1
        y = opt_y(stock + 1);
        [stock2, cost(j, i)] = per_week(y, r, stock);
        stock = stock2;
    end
    [cost(j, a), ~] = last_week(y, r, stock);

end

cost_pred = mean(sum(cost,2));

end