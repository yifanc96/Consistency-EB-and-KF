% test, L^2 norm approximation, expectation over the GP
% we calculate the expectation of L2 error using explicit formula

s=2.5;
fprintf('the ground truth is s=%g \n', s);
arr_q=9:10; num_q=length(arr_q);
arr_t=0.5:0.1:3; num_t=length(arr_t);
L2=zeros(num_t,num_q);
minimizer=zeros(num_q,1);
M0s=2*sum((1:10^4).^(-2*s));

for iter_q=1:num_q
    for iter_t=1:num_t
        q=arr_q(iter_q);
        t=arr_t(iter_t);
        a=Mtq(2*t,q); b=Mtq(s,q); c=Mtq(t,q); d=Mtq(t+s,q);
        L2(iter_t,iter_q)=M0s+sum(a.*b./(c.^2)-2*d./c);
    end
end

% calculate the intercept and slope of the linear regressor for the curves
slope_intercept=zeros(num_t,2); %1st element is slope, 2nd is intercept

for iter_t=1:num_t
    xi=arr_q;
    yi=log(L2(iter_t,:));
    regress_A=[sum(xi.^2),sum(xi);sum(xi),num_q];
    regress_rhs=[sum(xi.*yi);sum(yi)];
    regress_sol=regress_A\regress_rhs;
    slope_intercept(iter_t,:)=regress_sol';
end

figure;
plot(arr_t,slope_intercept(:,1));
xlabel('t'); ylabel('slope')
title('slope of the linear regressor for $(q,\mathbb{E}_u\|u-u(\cdot,t,q)\|_0)$, $q=3,4,...,9$','Interpreter','latex');

figure;
plot(arr_t,slope_intercept(:,2));
xlabel('t'); ylabel('intercept')
title('intercept of the linear regressor for $(q,\mathbb{E}_u\|u-u(\cdot,t,q)\|_0)$, $q=3,4,...,9$','Interpreter','latex');



function [M]=Mtq(t,q)
% compute M_{t,q} for all m \in B_q^d
m=-2^(q-1):2^(q-1)-1;
truncate_num=5*1e+3;
beta=-truncate_num:truncate_num; beta=beta';
tmp=abs(m+beta*2^(q)).^(-2*t);
tmp(tmp==inf)=0;
M=sum(tmp,1);
end

