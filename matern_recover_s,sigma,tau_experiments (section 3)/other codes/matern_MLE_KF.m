%variance of estimator
%Matern kernel
%all parameters together

N=2^9; hg=1/(N+1); %fine mesh N
t=(0:1:N)+0.5; v=kappa(t*hg)'; 
A=spdiags([-v(2:N+1),v(1:N)+v(2:N+1),-v(1:N)],-1:1,N,N)/hg^2; clear v

[eigvec,eigval]=eig(full(A)); %eigen-pairs of A
s=2.5;  %ground truth
sigma=0;
tau=0;
fprintf('ground truth for s,sigma,tau is %g, %g, %g\n',s,sigma,tau);
lambda=((diag(eigval)+exp(tau)).^s)*exp(sigma);

s_search=0:0.5:3; ls=length(s_search); %grid search of alpha
sigma_search=-1:0.5:1; lsig=length(sigma_search);
tau_search=-1:0.5:1; ltau=length(tau_search);
L_MLE=zeros(ls,lsig,ltau); L_KF=zeros(ls,lsig,ltau); %loss function

% array_log2n=1:9; ln=length(array_log2n); %n: num of data we observe; want to see the asymptotics when n goes to infinity
log2n=7; % num of data points 2^7

tot_iter=20;
sol_MLE=zeros(tot_iter,3); sol_KF=sol_MLE; % store estimators
testerr=zeros(tot_iter,2);
energy_err=zeros(tot_iter,2);
mtx_energy=eigvec*diag(1./diag(eigval.^1))*eigvec';


tic
for iter=1:tot_iter
    xi=randn(N,1); 
    u=eigvec*(xi./sqrt(lambda)); % generate samples of GP
    u=u+0*randn(N,1); %noise
    n=2^log2n;  %num of data observed
    index=1:N/n:N; indexs=1:2:n; %index of the observed data and subsampling
    Pid=sparse(1:n,index,ones(n,1),n,N);  %Pid is for the data
    Pis=sparse(1:n/2,indexs,ones(n/2,1),n/2,n); %Pis is for the subsampling
    u_data=Pid*u; u_sub=Pis*u_data;

    for iter_s=1:ls
        for iter_sig=1:lsig
            for iter_tau=1:ltau
                t=s_search(iter_s);
                sig=sigma_search(iter_sig);
                ta=tau_search(iter_tau);
                eigval_now=exp(sig)*(diag(eigval)+exp(ta)).^t;
                Theta_global=eigvec*diag(1./eigval_now)*eigvec';
                Theta_data=Pid*eigvec*diag(1./eigval_now)*eigvec'*Pid';
                Theta_sub=Pis*Theta_data*Pis';
                norm_data=u_data'*(Theta_data\u_data);
                norm_sub=u_sub'*(Theta_sub\u_sub);
                logdet=2*sum(log(diag(chol(Theta_data))));
                
                L_MLE(iter_s,iter_sig,iter_tau)=norm_data+logdet;
                L_KF(iter_s,iter_sig,iter_tau)=1-norm_sub/norm_data;
            end
        end
    end
    [i]=find(L_MLE(:)==min(L_MLE(:))); i=i(1);
    [j1,j2,j3]=ind2sub([ls,lsig,ltau],i);
    sol_MLE(iter,:)=[s_search(j1),sigma_search(j2),tau_search(j3)];
    eigval_now=exp(sig)*(diag(eigval)+exp(ta)).^t;
    Theta_now=eigvec*diag(1./eigval_now)*eigvec';
    u_recover=Theta_now*(Pid'*((Pid*(Theta_now*Pid'))\u_data));
    testerr(iter,1)=norm(u-u_recover)*sqrt(hg);
    energy_err(iter,1)=sqrt((u-u_recover)'*((mtx_energy)\(u-u_recover)));
    
    [i]=find(L_KF(:)==min(L_KF(:)));
    i=i(1);
    [j1,j2,j3]=ind2sub([ls,lsig,ltau],i);
    sol_KF(iter,:)=[s_search(j1),sigma_search(j2),tau_search(j3)];
    eigval_now=exp(sig)*(diag(eigval)+exp(ta)).^t;
    Theta_now=eigvec*diag(1./eigval_now)*eigvec';
    u_recover=Theta_now*(Pid'*((Pid*(Theta_now*Pid'))\u_data));
    testerr(iter,2)=norm(u-u_recover)*sqrt(hg);
    energy_err(iter,2)=sqrt((u-u_recover)'*((mtx_energy)\(u-u_recover)));
    fprintf('No. %g, data n=%g, MLE %g,%g,%g, KF %g,%g,%g, took %g s\n',iter,n,sol_MLE(iter,1),sol_MLE(iter,2),sol_MLE(iter,3),sol_KF(iter,1),sol_KF(iter,2),sol_KF(iter,3),toc);
end




function [y]=kappa(x)
    y=ones(size(x));
end