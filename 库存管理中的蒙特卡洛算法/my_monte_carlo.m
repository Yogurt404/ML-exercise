clc
clear all
% variable initialize
a = 52; % define 52 weeks
D = zeros(1,a);
estimate_num = 500; % number of simulations
cost = zeros(1,estimate_num);
stock = 0; % initial stock
penalty = 20;% short of stock penalty

% choose stock and re-order stock level
y = 3; % order number
r = 1; % re-order stock level

% Task 2 
for i = 1:estimate_num
    D = dw(a);
    for j = 1:a
        % check if need to re-order
        if stock <= r
            stock = stock + y; % stock after re-order
        end
        % check if inventory meets demand
        if stock >= D(1,j)
            stock = stock -D(1,j);
        else
            cost(1,i) = cost(1,i) + penalty;% short of stock penalty
            stock = 0;
        end
        % check if need warehouse cost
        if stock >0
            cost(1,i) = cost(1,i) + 5 * stock;% warehouse cost
        end
    end
    % check if need return cost
    if stock >0
        cost(1,i) = cost(1,i) + 10*stock;% return cost
    end
end
average_cost = sum(cost)/estimate_num% average of total cost after 500 estimates
var_cost = std(cost)% variance of total cost after 500 estimates
% plot Histogram of 500 estimates of the total cost at order number 3 and re-order stock level 1
figure()
histogram(cost,10)
xlabel('costs of 52 weeks')
ylabel('number of instance')
title('Histogram of 500 estimates of the total cost')


% Task 3
% variable initialize
clear all
a = 52;
D = zeros(1,a);
estimate_num = 50;
estimate_num1 = 1000;
stock = zeros(7,7);
opt_y = zeros(1,estimate_num);
opt_r = zeros(1,estimate_num);
opt_choice = zeros(estimate_num1,49);
y_index = 1;
r_index = 1;
combin = zeros(1,49);
average_cost = zeros(estimate_num1,49);
cost1 = zeros(estimate_num,49);

for i =1:estimate_num1% repeat 1000 times
    cost2=zeros(1,49);
    for j = 1:estimate_num
        cost = zeros(7,7);
        D = dw(a);
        for r = 0:6
            for y = 0:6
                for k = 1:a   
                    % import
                    if stock(y+1,r+1)<=r
                        stock(y+1,r+1) = stock(y+1,r+1) + y;
                    end
                    
                    % check if inventory meets demand
                    if stock(y+1,r+1)>=D(1,k)
                        stock(y+1,r+1) = stock(y+1,r+1) - D(1,k);
                    else
                        cost(y+1,r+1) = cost(y+1,r+1) + 20;
                       
                        stock(y+1,r+1) = 0;
                    end
                    
                    % check if need warehouse cost
                    if stock(y+1,r+1)>0
                        cost(y+1,r+1) = cost(y+1,r+1)+5*stock(y+1,r+1);
                        
                    end
                end
                if stock(y+1,r+1)>0
                    cost(y+1,r+1) = cost(y+1,r+1) + 10*stock(y+1,r+1);
                    
                end
            end
        end
%         if stock(y+1,r+1)>0
%             cost(y+1,r+1) = cost(y+1,r+1) + 10*stock(y+1,r+1);
%         end
        
        min_cost = min(min(cost));% for taking the smalest cost
        [row,column] = find(cost == min_cost);%Finding min cost value's row and column
        
        % produce optimal combination set
        for i1 = 1:length(row)
            opt_y(1,y_index) = row(i1) - 1;
            opt_r(1,r_index) = column(i1) - 1;
            y_index = y_index +1;
            r_index = r_index +1;
            index = 7*(row(i1)-1)+column(i1);
            opt_choice(i,index) = opt_choice(i,index) + 1;
        end
        
        % converte cost set from 2 choice to combination

        for i2 = 0:6
            for i3 = 0:6
                index = 7*i2+i3+1;
                cost1(j,index) = cost(i2+1,i3+1);
                cost2(1,index) = cost(i2+1,i3+1) + cost2(1,index);
                cost3 = cost2/estimate_num;
                total(index,:) = [cost3 i2 i3];
                
            end
        end
        
        
    end
    average_cost(i,:) = cost2/estimate_num;
end
mean_y = mean(opt_y);
var_y = std(opt_y);
mean_r = mean(opt_r);
var_r = std(opt_r);
figure()
histogram(opt_y)
hold on
histogram(opt_r)
title('Histogram of optimal order number and re-order threshold')
xlabel('order number/re-order threshold ')
ylabel('number of instance')
hold off
% probabilty of each choice combination
prob = zeros(estimate_num1,49);
row_value = zeros(1,49);
for i = 1:estimate_num1
    for j = 1:49
        prob(i,j) = opt_choice(i,j)/r_index;
        prob(i,j) = prob(i,j)*estimate_num1;
        row_value(:,j) = sum(prob(:,j));
    end
end
value_row_max = max(row_value);
row_max = find(row_value == value_row_max);
figure()
histogram(prob(:,row_max))
title('Histogram of probability of total cost of optimal combination')
xlabel('probability')
ylabel('number of instance')



% find low and high bound of cost for every combination
bound = zeros(2,49);
for i = 1:49
    bound(1,i) = min(average_cost(:,i));
    bound(2,i) = max(average_cost(:,i));
end
%*****************cost*****************
var = std(average_cost(:,row_max))
mean = mean(average_cost(:,row_max))
% prob_opt = zeros(2,49);
% for i = 1:49
%     if bound(1,i)<= bound(2,row_max)&& i~=row_max
%         prob_opt(1,i) = bound(1,i);
%         prob_opt(2,i) = bound(2,row_max)-bound(2,i);
%     end
% end
% 
% figure()
% hold on
% histogram(average_cost(:,row_max))
% % compare the lowest bound of other combination to the high bound of
% % optimal combination
% row_1 = find(prob_opt(1,:)~=0);
% for i = 1:length(row_1)
%     histogram(average_cost(:,row_1(i)))
% end
% hold off
% title('distribution of total cost of possible optimal combination')
% xlabel('total cost')
% ylabel('number of instance')
% legend('combination (3,1)','combination (3,1)')

%a = 2.58;
a = 1.645;
%*****************cost*****************
lower_bound_cost=mean-a*var/sqrt(500);%lower bound
upper_bound_cost=mean+a*var/sqrt(500);%upper bound
%*****************stock*****************
lower_bound_stock=round(mean_r)-a*round(var_r)/sqrt(500);%lower bound
upper_bound_stock=round(mean_r)+a*round(var_r)/sqrt(500);%upper bound
%*****************order number*****************
lower_bound_order=round(mean_y)-a*round(var_y)/sqrt(500);%lower bound
upper_bound_order=round(mean_y)+a*round(var_y)/sqrt(500);%upper bound       
        
Variables = {'ORDER NUMBER';'STOCK NUMBER';'MINIMUM COST'};

confidence_level = '90%';

CONFIDENCE_LEVELS = [confidence_level; confidence_level; confidence_level];
MEANS = [mean_y;mean_r;mean];
STANDART_DERIVATIONS = [var_y; var_r; var];
UPPER_BOUNDS = [upper_bound_order;upper_bound_stock;upper_bound_cost];
LOWER_BOUNDS = [lower_bound_order;lower_bound_stock;lower_bound_cost];

RESULTS = table(MEANS,STANDART_DERIVATIONS,CONFIDENCE_LEVELS,LOWER_BOUNDS,UPPER_BOUNDS,'RowNames',Variables); % create table to show all results
display(RESULTS)


% demand function
function y = dw(a)
for i = 1:a
    x = rand;
    if x<=0.04
        demand(1,i) = 0;
    elseif 0.04<x&x<=0.12
        demand(1,i)=1;
    elseif 0.12<x&x<=0.4
        demand(1,i)=2;
    elseif 0.4<x&x<=0.8
        demand(1,i)=3;
    elseif 0.8<x&x<=0.96
        demand(1,i)=4;
    elseif 0.96<x&x<=0.98
        demand(1,i)=5;
    elseif 0.98<x&x<=1
        demand(1,i)=6;
    end
end
y = demand;
end