% general conductivity field a
%recover s for EB, KF loss
%section 2 and 3 in the paper

clear
N=2^10; hg=1/(N+1); %fine mesh N
t=(0:1:N)+0.5; v=kappa(t*hg)'; 
A=spdiags([-v(2:N+1),v(1:N)+v(2:N+1),-v(1:N)],-1:1,N,N)/hg^2; clear v
v_lap=kappa_lap(t*hg)';
A_lap=spdiags([-v(2:N+1),v(1:N)+v(2:N+1),-v(1:N)],-1:1,N,N)/hg^2;

[eigvec,eigval]=eig(full(A)); %eigen-pairs of A
s=2.5;  %ground truth
fprintf('ground truth for s is %g\n',s);
lambda=(diag(eigval).^s);
xi=randn(N,1); u=eigvec*(xi./sqrt(lambda)); % generate samples of GP

s_search=0.5:0.025:3; la=length(s_search); %grid search of s

array_log2n=8:8; ln=length(array_log2n); %n: num of data we observe; want to see the asymptotics when n goes to infinity
sol_MLE=zeros(ln,1); sol_KF=sol_MLE; % store estimators
testerr=zeros(ln,2);

L_MLE=zeros(la,ln); L_KF=zeros(la,ln); %loss function
tic
[eigvec,eigval]=eig(full(A_lap)); %mis-specification, use eigenfunction of laplacian for GP modeling
for iter=1:ln
    n=2^array_log2n(iter);  %num of data observed
    index=1:N/n:N; indexs=1:2:n; %index of the observed data and subsampling
    Pid=sparse(1:n,index,ones(n,1),n,N);  %Pid is for the data
    Pis=sparse(1:n/2,indexs,ones(n/2,1),n/2,n); %Pis is for the subsampling
    u_data=Pid*u; u_sub=Pis*u_data;

    for iter_a=1:la
        a=s_search(iter_a);
        Theta_global=eigvec*diag(1./diag(eigval.^a))*eigvec';
        Theta_data=Pid*eigvec*diag(1./diag(eigval.^a))*eigvec'*Pid';
        Theta_sub=Pis*Theta_data*Pis';
        norm_data=u_data'*(Theta_data\u_data);
        norm_sub=u_sub'*(Theta_sub\u_sub);
        logdet=2*sum(log(diag(chol(Theta_data))));
        
        L_MLE(iter_a,iter)=norm_data+logdet;
        L_KF(iter_a,iter)=1-norm_sub/norm_data;
    end
    
    [i]=find(L_MLE(:,iter)==min(L_MLE(:,iter)));
    sol_MLE(iter)=s_search(i(1));
    
    [i]=find(L_KF(:,iter)==min(L_KF(:,iter)));
    sol_KF(iter)=s_search(i(1));
    fprintf('No. %g, data n=%g, MLE %g, KF %g, took %g s\n',iter,n,sol_MLE(iter),sol_KF(iter),toc);
end

% plot figures for test
figure
h=zeros(1,2);
h(1)=plot(s_search,L_MLE(:,end));
title('Loss function: Empirical Bayesian');
xlabel('t'); ylabel('loss');
hold on
i=find(L_MLE==min(L_MLE));
h(2)=plot(s_search(i),L_MLE(i,end),'r*');
legend(h(2),'minimizer');

figure;
h2=zeros(1,2);
h2(1)=semilogy(s_search,L_KF(:,end));
title('Loss function: Kernel Flow');
xlabel('t'); ylabel('loss');
hold on
i=find(L_KF==min(L_KF));
h2(2)=plot(s_search(i),L_KF(i,end),'r*');
legend(h2(2),'minimizer');

% plot figures for paper
% save data_recover_s s_search L_MLE L_KF
% Figplot_recover_s;


function [y]=kappa(x)
    a=1/2; b=2;
    y=(x<a).*ones(size(x))+(x>=a).*b.*ones(size(x));
end

function [y]=kappa_lap(x)
    y=ones(size(x));
end

