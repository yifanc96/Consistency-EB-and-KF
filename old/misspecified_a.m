clear
N=2^10; hg=1/(N+1); %fine mesh N
t=(0:1:N)+0.5; v=kappa(t*hg)'; 
A=spdiags([-v(2:N+1),v(1:N)+v(2:N+1),-v(1:N)],-1:1,N,N)/hg^2; clear v
% A(1,end)=A(1,2); A(end,1)=A(end,end-1);
% A=sparse(A);

[eigvec,eigval]=eig(full(A)); %eigen-pairs of A
alpha=2.5;  %ground truth
fprintf('ground truth for alpha is %g\n',alpha);
lambda=(diag(eigval).^alpha);

alpha_search=0.5:0.025:3; la=length(alpha_search); %grid search of alpha

array_log2n=8:8; ln=length(array_log2n); %n: num of data we observe; want to see the asymptotics when n goes to infinity

L_MLE=zeros(la,ln); L_KF=zeros(la,ln); %loss function
tic
num_sample=1;
sol_MLE=zeros(num_sample,ln); sol_KF=sol_MLE; % store estimators

% t=(0:1:N)+0.5; v2=kappa2(t*hg)'; 
% A2=spdiags([-v2(2:N+1),v2(1:N)+v2(2:N+1),-v2(1:N)],-1:1,N,N)/hg^2;
% [eigvec2,eigval2]=eig(full(A2));
% lambda2=(diag(eigval2).^alpha);
for iter_sample=1:num_sample
%     xi=randn(N,1); 
%     u=eigvec2*(xi./sqrt(lambda2));
%     u=sqrt(hg)*(A\[1;zeros(N/2,1);zeros(N/2-1,1)]);
%     u=sqrt(hg)*(A\ones(N,1));
    u=eigvec*diag(1./(diag(eigval).^(0.8)))*eigvec'*[zeros(N/2,1);1;zeros(N/2-1,1)];
    for iter=1:ln
        n=2^array_log2n(iter);  %num of data observed
        index=1:N/n:N; indexs=1:2:n; %index of the observed data and subsampling
        Pid=sparse(1:n,index,ones(n,1),n,N);  %Pid is for the data
        Pis=sparse(1:n/2,indexs,ones(n/2,1),n/2,n); %Pis is for the subsampling
        u_data=Pid*u; u_sub=Pis*u_data;
        
        for iter_a=1:la
            a=alpha_search(iter_a);
%             Theta_global=eigvec*diag(1./diag(eigval.^a))*eigvec';
            Theta_data=Pid*eigvec*diag(1./diag(eigval.^a))*eigvec'*Pid'/hg;
            Theta_sub=Pis*Theta_data*Pis';
            norm_data=u_data'*(Theta_data\u_data);
            norm_sub=u_sub'*(Theta_sub\u_sub);
            logdet=2*sum(log(diag(chol(Theta_data))));
            
            L_MLE(iter_a,iter)=norm_data+logdet;
            L_KF(iter_a,iter)=1-norm_sub/norm_data;
        end
        
        [i]=find(L_MLE(:,iter)==min(L_MLE(:,iter)));
        sol_MLE(iter_sample,iter)=alpha_search(i(1));
        
        [i2]=find(L_KF(:,iter)==min(L_KF(:,iter)));
        sol_KF(iter_sample, iter)=alpha_search(i2(1));
        fprintf('No. %g, data n=%g, MLE %g, KF %g, took %g s\n',iter,n,sol_MLE(iter_sample,iter),sol_KF(iter_sample,iter),toc);
    end
end


% figure;
% hist_ax=2.3:0.05:2.7;
% temp=hist(sol_MLE,hist_ax);
% bar(hist_ax,temp)
% set(gca,'XTick',2.3:0.05:2.7 );
% xlabel('s');
% 
% figure;
% hist_ax=0.8:0.05:1.2;
% temp=hist(sol_KF,hist_ax);
% bar(hist_ax,temp)
% set(gca,'XTick',0.8:0.05:1.2);
% xlabel('s');

figure
h=zeros(1,2);
h(1)=plot(alpha_search,L_MLE(:,end));
title('Loss function: Empirical Bayesian');
xlabel('t'); ylabel('loss');
xticks([0.5:0.5:2.5,2.6,3])
hold on
h(2)=plot(alpha_search(i(1)),L_MLE(i(1),end),'r*');
legend(h(2),'minimizer');

figure;
h2=zeros(1,2);
h2(1)=semilogy(alpha_search,L_KF(:,end));
title('Loss function: Kernel Flow');
xlabel('t'); ylabel('loss');
xticks([0.5,0.8,1.5:0.5:3])
hold on
h2(2)=plot(alpha_search(i2(1)),L_KF(i2(1),end),'r*');
legend(h2(2),'minimizer');

function [y]=kappa(x)
    y=ones(size(x));
end

function [y]=kappa2(x)
    a=0.5;b=2;
    y=(x<a).*ones(size(x))+(x>=a).*b.*ones(size(x));
%     y=exp(-(x-a).^2./b)+1;
%     y=3*exp(-(x-a).^2./b).*(x<a)+3*(x>=a)+1;
%     tmp1=exp(-0.001./(x-a).^2).*(x-a>0);
%     tmp2=exp(-0.001./(b-x+a).^2).*(b-x+a>0);
%     y=(tmp1./(tmp1+tmp2))+1;
end

