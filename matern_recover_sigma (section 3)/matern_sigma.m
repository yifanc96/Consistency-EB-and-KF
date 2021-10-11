
%Matern kernel
%recover sigma for EB loss
%section 3 in the paper

d=1;
N=2^10; hg=1/(N+1); %fine mesh N
t=(0:1:N)+0.5; v=kappa(t*hg)'; 
A=spdiags([-v(2:N+1),v(1:N)+v(2:N+1),-v(1:N)],-1:1,N,N)/hg^2; clear v
[eigvec,eigval]=eig(full(A)); %eigen-pairs of A

s=2.5;  %ground truth
sigma=0;
tau=0;
fprintf('ground truth for s,sigma,tau is %g, %g, %g\n',s,sigma,tau);

lambda=((diag(eigval)+exp(tau)).^s)*exp(sigma);
s_search=2.5; ls=length(s_search); 
sigma_search=-1.5:0.01:1.5; lsig=length(sigma_search); %grid search of sigma
tau_search=0; ltau=length(tau_search);
L_MLE=zeros(ls,lsig,ltau); %loss function

log2n=8; % num of data points 2^8

tot_iter=1; % only one random instance
sol_MLE=zeros(tot_iter,3);  % store estimators

tic
for iter=1:tot_iter
    xi=randn(N,1); 
    u=eigvec*(xi./sqrt(lambda)); % generate samples of GP
    n=2^log2n;  %num of data observed
    index=1:N/n:N; indexs=1:2:n; %index of the observed data and subsampling
    Pid=sparse(1:n,index,ones(n,1),n,N);  %Pid is for the data
    u_data=Pid*u; 
    for iter_s=1:ls
        for iter_sig=1:lsig
            for iter_tau=1:ltau
                t=s_search(iter_s);
                sig=sigma_search(iter_sig);
                ta=tau_search(iter_tau);
                eigval_now=exp(sig)*(diag(eigval)+exp(ta)).^t;
                Theta_global=eigvec*diag(1./eigval_now)*eigvec';
                Theta_data=Pid*eigvec*diag(1./eigval_now)*eigvec'*Pid';
                norm_data=u_data'*(Theta_data\u_data);
                logdet=2*sum(log(diag(chol(Theta_data))));
                
                L_MLE(iter_s,iter_sig,iter_tau)=norm_data+logdet;
            end
        end
    end
    [i]=find(L_MLE(:)==min(L_MLE(:)));
    [j1,j2,j3]=ind2sub([ls,lsig,ltau],i);
    sol_MLE(iter,:)=[s_search(j1),sigma_search(j2),tau_search(j3)];
    fprintf('No. %g, data n=%g, MLE %g,%g,%g, took %g s\n',iter,n,sol_MLE(iter,1),sol_MLE(iter,2),sol_MLE(iter,3),toc);
end

% plot figures for test
figure
h=zeros(1,2);
h(1)=plot(sigma_search,L_MLE);
i=find(L_MLE==min(L_MLE));
title('Loss function: Empirical Bayesian');
xlabel('log \varsigma'); ylabel('loss');
hold on
h(2)=plot(sigma_search(i),L_MLE(i),'r*');
legend(h(2),'minimizer');


% plot the figures for paper
% save data_recover_sigma sigma_search L_MLE
% Figplot_recover_sigma;



function [y]=kappa(x)
    y=ones(size(x));
end