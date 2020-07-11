%% Gain optimal y and check if need re-order
function [opt_y, order_flag] = gain_opt_y(y, r, h_ctrl,h_pred, estimate_num, stock)
% Initialization
y2 = combvec(y, y, y, y, y, y, y);
cost = zeros(7 ^ h_ctrl, 1);

% Cost prediction
for i = 1: size(cost, 1)

    cost(i) = pred_cost(stock, h_pred, h_ctrl, y2(:, i), r, estimate_num);
end
[~, idx] = min(cost);

%Check if need re-order
if stock > r
    order_flag = 0;
else
    order_flag = 1;
end


% Gain optimal y
if order_flag == 1
    opt_y = y2(1, idx);
    if h_ctrl == 1 && h_pred == 1
    opt_y = max(3 - stock, 0);
    end

else
    opt_y = 0;
end

end