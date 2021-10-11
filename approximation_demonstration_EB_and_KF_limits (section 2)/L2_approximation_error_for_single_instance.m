clear
N=2^10; hg=1/(N+1); %fine mesh N
t=(0:1:N)+0.5; v=kappa(t*hg)'; 
A=spdiags([-v(2:N+1),v(1:N)+v(2:N+1),-v(1:N)],-1:1,N,N)/hg^2; clear v

[eigvec,eigval]=eig(full(A)); %eigen-pairs of A
s=2.5;  %ground truth
fprintf('ground truth for alpha is %g\n',s);
lambda=(diag(eigval).^s);

xi=randn(N,1); u=eigvec*(xi./sqrt(lambda)); % generate samples of GP
% u=eigvec*(ones(N,1)./sqrt(lambda)); % deterministic 

s_search=0.5:0.25:2.75; la=length(s_search); %grid search of s
array_log2n=4:9; ln=length(array_log2n); 
%n: num of data we observe
L2=zeros(la,ln);  %loss function

% store estimators
sol_L2=zeros(ln,1);

tic
for iter=1:ln
    n=2^array_log2n(iter);  %num of data observed
    index=1:N/n:N; indexs=1:2:n; %index of the observed data and subsampling
    Pid=sparse(1:n,index,ones(n,1),n,N);  %Pid is for the data
    u_data=Pid*u; 

    for iter_a=1:la
        a=s_search(iter_a);
        Theta_global=eigvec*diag(1./diag(eigval.^a))*eigvec';
        Theta_data=Pid*Theta_global*Pid';
        temp=Theta_data\u_data;
        u_interp=Theta_global*Pid'*(temp);
        norm_data=u_data'*(temp);

        L2(iter_a,iter)=norm(u-u_interp)*hg;
    end
    
    [i]=find(L2(:,iter)==min(L2(:,iter)));
    sol_L2(iter)=s_search(i(1));
    
    fprintf('No. %g, data n=%g, L2 %g, took %g s\n',iter,n, sol_L2(iter), toc);
end

figure;
for iter_plot=1:length(s_search)
    semilogy(array_log2n,L2(iter_plot,:)','Color',rand(1,3));
    hold on
end
legend('t=0.5','t=0.75','t=1','t=1.25','t=1.5','t=1.75','t=2','t=2.25','t=2.5','t=2.75');
xlabel('q');
ylabel('L^2 error');

% calculate the intercept and slope of the linear regressor for the curves
% slope_intercept=zeros(la,2);
% for iter_a=1:la
%     xi=array_log2n;
%     yi=log(L2(iter_a,:))/log(10);
%     regress_A=[sum(xi.^2),sum(xi);sum(xi),ln];
%     regress_rhs=[sum(xi.*yi);sum(yi)];
%     regress_sol=regress_A\regress_rhs;
%     slope_intercept(iter_a,:)=regress_sol';
% end


function [y]=kappa(x)
    y=ones(size(x));
end

