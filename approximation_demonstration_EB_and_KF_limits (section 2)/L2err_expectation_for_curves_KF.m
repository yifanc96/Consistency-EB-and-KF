% test, L^2 norm approximation, expectation over the GP
% demonstration that KF s-d/2 / 2 is the minimal t such that optimal
% convergence rate of L2 error occurs

s=2.5;
fprintf('the ground truth is s=%g \n', s);
arr_q=4:9; num_q=length(arr_q);
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
    fprintf('For q=%g, the minimizer w.r.t. t is %g\n',q, minimizer(iter_q));
end

% cmap = jet(num_t);
cmap=rand(num_t,3);
% semilogy(arr_q',L','color',cmap);
set(0,'defaultaxesfontsize',18,'defaultaxeslinewidth',1.2,...
    'defaultlinelinewidth',1.2,'defaultpatchlinewidth',1.2)
for i=1:num_t
    semilogy(arr_q,L(i,:),'color',cmap(i,:),'LineWidth',1.2);
    hold on
end
legend('t=0.5','t=0.75','t=1','t=1.25','t=1.5','t=1.75','t=2','t=2.25','t=2.5','t=2.75');
xlabel('q');
ylabel('L^2 error: averaged over the GP');

h=gcf;
myprint('L2_average_over_GP_for_KF_demonstration',h)

function [M]=Mtq(t,q)
% compute M_{t,q} for all m \in B_q^d
m=-2^(q-1):2^(q-1)-1;
beta=-40:40; beta=beta';
tmp=abs(m+beta*2^(q)).^(-2*t);
tmp(tmp==inf)=0;
M=sum(tmp,1);
end

