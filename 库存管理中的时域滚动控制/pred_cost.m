%% Cost prediction by prediction and control horizion
function  pred_cost = pred_cost(stock, h_ctrl,h_pred ,y, r, estimate_num)

for i = 1: estimate_num
    stock1 = stock;
    for j = 1: h_pred
        if j <= h_ctrl
            y1 = y(j);
        else
            y1 = y(h_ctrl);
        end
        [cost, stock2] = per_week(y1, r, stock1);
        pred_cost(i, j) = cost;
        stock1 = stock2;
    end
end

pred_cost = mean(sum(pred_cost, 2));

end