clear

%% stiffness mtx
N=2^10; hg=1/(N+1); %fine mesh N
t=(0:1:N)+0.5; v=kappa(t*hg)'; 
A=spdiags([-v(2:N+1),v(1:N)+v(2:N+1),-v(1:N)],-1:1,N,N)/hg^2; clear v
[eigvec,eigval]=eig(full(A)); %eigen-pairs of A
s=2.5;  %ground truth regularity parameter
fprintf('ground truth for s is %g\n',s);

% generate samples
lambda=(diag(eigval).^s);
% xi=randn(N,1);
xi=ones(N,1);
u=eigvec*(xi./sqrt(lambda)); % generate samples of GP

% searched value for s
s_search=0.5:0.25:3; ls=length(s_search); 

% array of the resolution parameter q
array_q=6:8; lq=length(array_q); %n: num of data we observe
L2=zeros(ls,lq);  %L2 error

tic
for iter_q=1:lq
    q=2^array_q(iter_q);  %num of data observed
    index=1:N/q:N; indexs=1:2:q; %index of the observed data
    Pid=sparse(1:q,index,ones(q,1),q,N);  %Pid is the sampling mtx for the data
    u_data=Pid*u; % the data observed

    for iter_s=1:ls
        t=s_search(iter_s);
        % calculate the GP mean function
        Theta_global=eigvec*diag(1./diag(eigval.^t))*eigvec'; %Gram mtx for the discretization
        Theta_data=Pid*Theta_global*Pid'; % Gram mtx for the data location
        temp=Theta_data\u_data;
        u_interp=Theta_global*Pid'*(temp); % GP mean function

        L2(iter_s,iter_q)=norm(u-u_interp)*hg; % L2 error
    end
end
toc

% calculate the intercept and slope of the linear regressor for the curves
slope_intercept=zeros(ls,2); %1st element is slope, 2nd is intercept

for iter_s=1:ls
    xi=array_q;
    yi=log(L2(iter_s,:));
    regress_A=[sum(xi.^2),sum(xi);sum(xi),lq];
    regress_rhs=[sum(xi.*yi);sum(yi)];
    regress_sol=regress_A\regress_rhs;
    slope_intercept(iter_s,:)=regress_sol';
end

figure;
plot(s_search,slope_intercept(:,1));
xlabel('t'); ylabel('slope')
title('slope of the linear regressor for $(q,\|u-u(\cdot,t,q)\|_0)$, $q=3,4,...,9$','Interpreter','latex');

figure;
plot(s_search,slope_intercept(:,2));
xlabel('t'); ylabel('intercept')
title('intercept of the linear regressor for $(q,\|u-u(\cdot,t,q)\|_0)$, $q=3,4,...,9$','Interpreter','latex');


function [y]=kappa(x)
    y=ones(size(x));
end

