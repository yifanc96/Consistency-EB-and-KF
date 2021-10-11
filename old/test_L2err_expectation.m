% test, L^2 norm approximation, expectation series

s=2.5;
arr_q=4:9; num_q=length(arr_q);
% arr_t=2:0.01:3;
arr_t=0.5:0.25:2.75;
num_t=length(arr_t);
L=zeros(num_t,num_q);
minimizer=zeros(num_q,1);
M0s=2*sum((1:10^4).^(-2*s));

for iter_q=1:num_q
    for iter_t=1:num_t
        q=arr_q(iter_q);
        t=arr_t(iter_t);
        a=Mtq(2*t,q); b=Mtq(s,q); c=Mtq(t,q); d=Mtq(t+s,q);
        L(iter_t,iter_q)=M0s+sum(a.*b./(c.^2)-2*d./c);
    end
    [i]=find(L(:,iter_q)==min(L(:,iter_q)));
    minimizer(iter_q)=arr_t(i(1));
end
disp(minimizer);
    
% figure
% h(1)=semilogy(arr_t,L);
% xlabel('t');
% ylabel('L^2 error: averaged over the GP');
% hold on
% i=find(L(:,end)==min(L(:,end)));
% h(2)=plot(arr_t(i),L(i,end),'r*');
% legend(h(2),'minimizer');

plot(arr_q,L);

function [M]=Mtq(t,q)
% compute M_{t,q} for all m \in B_q^d
m=-2^(q-1):2^(q-1)-1;
beta=-40:40; beta=beta';
tmp=abs(m+beta*2^(q)).^(-2*t);
tmp(tmp==inf)=0;
M=sum(tmp,1);
end

