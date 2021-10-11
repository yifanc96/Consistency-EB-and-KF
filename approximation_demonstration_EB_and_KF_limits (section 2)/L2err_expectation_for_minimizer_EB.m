% test, L^2 norm approximation, expectation series
% demonstration for EB achieves the best error in L2 sense

s=2.5;
arr_q=9; num_q=length(arr_q);
arr_t=2:0.01:3;
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
end

figure
h(1)=semilogy(arr_t,L);
xlabel('t');
ylabel('L^2 error: averaged over the GP');
hold on
i=find(L(:,end)==min(L(:,end)));
h(2)=plot(arr_t(i),L(i,end),'r*');
legend(h(2),'minimizer');
axesfontsize=16;
axeslinewidth=1.8;
linelinewidth=1.8;
patchlinewidth=1.5;
set(0,'defaultaxesfontsize',axesfontsize,'defaultaxeslinewidth',axeslinewidth,...
    'defaultlinelinewidth',linelinewidth,'defaultpatchlinewidth',patchlinewidth)
set(gca,'XTick',2:0.1:3);
h=gcf;
myprint('L2_average_over_GP_for_EB_demonstration',h)




function [M]=Mtq(t,q)
% compute M_{t,q} for all m \in B_q^d
m=-2^(q-1):2^(q-1)-1;
beta=-40:40; beta=beta';
tmp=abs(m+beta*2^(q)).^(-2*t);
tmp(tmp==inf)=0;
M=sum(tmp,1);
end

