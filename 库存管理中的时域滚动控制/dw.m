function y = dw(a)
demand = zeros(a,1);
for i = 1:a
    x = rand;
    if x<=0.04
        demand(i,1) = 0;
    elseif 0.04<x&x<=0.12
        demand(i,1)=1;
    elseif 0.12<x&x<=0.4
        demand(i,1)=2;
    elseif 0.4<x&x<=0.8
        demand(i,1)=3;
    elseif 0.8<x&x<=0.96
        demand(i,1)=4;
    elseif 0.96<x&x<=0.98
        demand(i,1)=5;
    elseif 0.98<x&x<=1
        demand(i,1)=6;
    end
end
y = demand;
end