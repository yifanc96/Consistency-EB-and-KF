% variance of estimator
% variable coefficient elliptic
% recover s

d=1;
N=2^10; hg=1/(N+1); %fine mesh N
t=(0:1:N)+0.5; v=kappa(t*hg)'; 
A=spdiags([-v(2:N+1),v(1:N)+v(2:N+1),-v(1:N)],-1:1,N,N)/hg^2; clear v
v_lap=kappa_lap(t*hg)';
A_lap=spdiags([-v_lap(2:N+1),v_lap(1:N)+v_lap(2:N+1),-v_lap(1:N)],-1:1,N,N)/hg^2;

[eigvec_A,eigval_A]=eig(full(A)); %eigen-pairs of A
s=2.5;  %ground truth
sigma=0;
tau=0;
fprintf('ground truth for s,sigma,tau is %g, %g, %g\n',s,sigma,tau);
lambda=((diag(eigval_A)+exp(tau)).^s)*exp(sigma);

s_search=0.8:0.01:2.7; ls=length(s_search); %grid search of alpha
sigma_search=0; lsig=length(sigma_search);
tau_search=0; ltau=length(tau_search);
L_MLE=zeros(ls,lsig,ltau); L_KF=zeros(ls,lsig,ltau); %loss function

% array_log2n=1:9; ln=length(array_log2n); %n: num of data we observe; want to see the asymptotics when n goes to infinity
log2n=8; % num of data points 

tot_iter=50; % total instances
sol_MLE=zeros(tot_iter,3); sol_KF=sol_MLE; % store estimators

tic
[eigvec,eigval]=eig(full(A_lap)); %mis-specification, use eigenfunction of laplacian for GP modeling
for iter=1:tot_iter
    xi=randn(N,1); 
    u=eigvec_A*(xi./sqrt(lambda)); % generate samples of GP
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
    [i]=find(L_MLE(:)==min(L_MLE(:)));
    [j1,j2,j3]=ind2sub([ls,lsig,ltau],i);
    sol_MLE(iter,:)=[s_search(j1),sigma_search(j2),tau_search(j3)];

    [i]=find(L_KF(:)==min(L_KF(:)));
    i=i(1);
    [j1,j2,j3]=ind2sub([ls,lsig,ltau],i);
    sol_KF(iter,:)=[s_search(j1),sigma_search(j2),tau_search(j3)];
    fprintf('No. %g, data n=%g, MLE %g,%g,%g, KF %g,%g,%g, took %g s\n',iter,n,sol_MLE(iter,1),sol_MLE(iter,2),sol_MLE(iter,3),sol_KF(iter,1),sol_KF(iter,2),sol_KF(iter,3),toc);
end

MLE_err_s=sol_MLE(:,1)-s;
MLE_ave_err_s=sum((MLE_err_s).^2)/tot_iter;
KF_err_s=sol_KF(:,1)-(s-d/2)/2;
KF_ave_err_s=sum(KF_err_s.^2)/tot_iter;
fprintf('MLE ave squared err %g, KF ave squared err %g\n',MLE_ave_err_s,KF_ave_err_s);

r_MLE_ave_err_s=MLE_ave_err_s/s^2;
r_KF_ave_err_s=KF_ave_err_s/((s-d/2)^2);
fprintf('normalized MLE ave squared err %g, KF ave squared err %g',r_MLE_ave_err_s,r_KF_ave_err_s);

% plot histogram figures
figure
hist_ax=2.3:0.04:2.7;
temp=hist(sol_MLE(:,1),hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',2.3:0.04:2.7);
xlabel('s^{EB}');

figure
hist_ax=0.8:0.04:1.2;
temp=hist(sol_KF(:,1),hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',0.8:0.04:1.2);
xlabel('s^{KF}');


% save data_variance_misspecify sol_KF sol_MLE  MLE_ave_err_s KF_ave_err_s

function [y]=kappa(x)
    a=1/2; b=2;
    y=(x<a).*ones(size(x))+(x>=a).*b.*ones(size(x));
end

function [y]=kappa_lap(x)
    y=ones(size(x));
end