
%Matern kernel
%recover tau

d=1;
N=2^9; hg=1/(N+1); %fine mesh N
t=(0:1:N)+0.5; v=kappa(t*hg)'; 
A=spdiags([-v(2:N+1),v(1:N)+v(2:N+1),-v(1:N)],-1:1,N,N)/hg^2; clear v

[eigvec,eigval]=eig(full(A)); %eigen-pairs of A
s=2.5;  %ground truth
sigma=0;
tau=0;
fprintf('ground truth for s,sigma,tau is %g, %g, %g\n',s,sigma,tau);
lambda=((diag(eigval)+exp(tau)).^s)*exp(sigma);

s_search=2.5; ls=length(s_search); %grid search of alpha
sigma_search=0; lsig=length(sigma_search);
tau_search=-2:0.05:2; ltau=length(tau_search);
L_MLE=zeros(ls,lsig,ltau); L_KF=zeros(ls,lsig,ltau); %loss function
L_KF2=zeros(ls,lsig,ltau); %KF2 corresponds to take t=(s-d/2)/2

% array_log2n=1:9; ln=length(array_log2n); %n: num of data we observe; want to see the asymptotics when n goes to infinity
log2n=8; % num of data points 2^7

tot_iter=50;
sol_MLE=zeros(tot_iter,3); sol_KF=sol_MLE; sol_KF2=sol_KF;% store estimators


tic
for iter=1:tot_iter
    xi=randn(N,1); 
    u=eigvec*(xi./sqrt(lambda)); % generate samples of GP
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
                
                eigval_now2=exp(sig)*(diag(eigval)+exp(ta)).^((t-d/2)/2);
                Theta_global2=eigvec*diag(1./eigval_now2)*eigvec';
                Theta_data2=Pid*eigvec*diag(1./eigval_now2)*eigvec'*Pid';
                Theta_sub2=Pis*Theta_data2*Pis';
                norm_data2=u_data'*(Theta_data2\u_data);
                norm_sub2=u_sub'*(Theta_sub2\u_sub);
                
                L_MLE(iter_s,iter_sig,iter_tau)=norm_data+logdet;
                L_KF(iter_s,iter_sig,iter_tau)=1-norm_sub/norm_data;
                L_KF2(iter_s,iter_sig,iter_tau)=1-norm_sub2/norm_data2;
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
    
    [i]=find(L_KF2(:)==min(L_KF2(:)));
    i=i(1);
    [j1,j2,j3]=ind2sub([ls,lsig,ltau],i);
    sol_KF2(iter,:)=[s_search(j1),sigma_search(j2),tau_search(j3)];

    
    fprintf('No. %g, data n=%g, MLE %g,%g,%g, KF %g,%g,%g, took %g s\n',iter,n,sol_MLE(iter,1),sol_MLE(iter,2),sol_MLE(iter,3),sol_KF(iter,1),sol_KF(iter,2),sol_KF(iter,3),toc);
end

% L_MLE=L_MLE(:);
% L_KF=L_KF(:);
% L_KF2=L_KF2(:);
% figure;
% h=zeros(1,2);
% h(1)=plot(tau_search,L_MLE);
% i=find(L_MLE==min(L_MLE));
% title('Loss function: Empirical Bayesian');
% xlabel('log tau'); ylabel('loss');
% hold on
% h(2)=plot(tau_search(i),L_MLE(i),'r*');
% legend(h(2),'minimizer');
% 
% 
% figure;
% h=zeros(1,2);
% h(1)=plot(tau_search,L_KF);
% i=find(L_KF==min(L_KF));
% title('Loss function: Kernel Flow (case 1)');
% xlabel('log tau'); ylabel('loss');
% hold on
% h(2)=plot(tau_search(i),L_KF(i),'r*');
% legend(h(2),'minimizer');
% 
% 
% figure;
% h=zeros(1,2);
% h(1)=plot(tau_search,L_KF2);
% i=find(L_KF2==min(L_KF2));
% title('Loss function: Kernel Flow (case 2)');
% xlabel('log tau'); ylabel('loss');
% hold on
% h(2)=plot(tau_search(i),L_KF2(i),'r*');
% legend(h(2),'minimizer');

% draw histograms
% need to increase the tot_iteration

%save data
% save data_matern_tau sol_MLE sol_KFu sol_KF2

sol_MLE_tau=sol_MLE(:,3);
figure;
hist_ax=-2:0.25:2;
temp=hist(sol_MLE_tau,hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',-2:0.25:2 );
xlabel('log \tau^{EB}');

sol_KF_tau=sol_KF(:,3);
figure;
hist_ax=-2:0.25:2;
temp=hist(sol_KF_tau,hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',-2:0.25:2 );
xlabel('log \tau^{KF}');

sol_KF2_tau=sol_KF2(:,3);
figure;
hist_ax=-2:0.25:2;
temp=hist(sol_KF2_tau,hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',-2:0.25:2 );
xlabel('log \tau^{KF}');

function [y]=kappa(x)
    y=ones(size(x));
end