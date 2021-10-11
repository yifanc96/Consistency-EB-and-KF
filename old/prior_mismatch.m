clear
N=2^10; hg=1/(N+1);
a_true=N/2*hg; b_true=2;
t=(0:1:N)+0.5; v=kappa(t*hg,a_true,b_true)'; 
fprintf('ground truth for a,b is %g, %g\n',a_true,b_true);
A=spdiags([-v(2:N+1),v(1:N)+v(2:N+1),-v(1:N)],-1:1,N,N)/hg^2; clear v

[eigvec,eigval]=eig(full(A)); %eigen-pairs of A
lambda=diag(eigval).^(4);
xi=randn(N,1);
u=1/sqrt(hg)*eigvec*(xi./sqrt(lambda)); % generate samples

% eigval=diag(pi^2*(1:N).^2);
% tgrid=(1:N)*hg; tkgrid=bsxfun(@times,tgrid',pi*(1:N));
% % eigvec=sin(tkgrid)*sqrt(2)*sqrt(hg);
% eigvec=sin(tkgrid)*sqrt(2);
% alpha=0.5;
% lambda=diag(eigval).^alpha; rhs_lap=eigvec*diag(lambda)*eigvec';
% xi=randn(N,1); 
% u=A\(eigvec*(xi./sqrt(lambda)));

a_search=0.3:0.01:0.7; la=length(a_search);
b_search=b_true; lb=length(b_search);
L_MLE=zeros(la,lb); L_KF=zeros(la,lb); %loss function
array_log2n=1:10; ln=length(array_log2n); %n: num of data
sol_MLE=zeros(ln,2); sol_KF=sol_MLE; %estimators
testerr=zeros(ln,2);
relenergy=zeros(ln,2);

tic
for iter=ln-2:ln
    n=2^array_log2n(iter);
    index=1:N/n:N; indexs=2:2:n;
    Pid=sparse(1:n,index,ones(n,1),n,N);  %Pid is the data
    Pis=sparse(1:n/2,indexs,ones(n/2,1),n/2,n); %Pis is the subsampling
    u_data=Pid*u; u_sub=Pis*u_data;

    for iter_a=1:la
        for iter_b=1:lb
            a=a_search(iter_a); b=b_search(iter_b);
            v=kappa(t*hg,a,b)';
            Aab=spdiags([-v(2:N+1),v(1:N)+v(2:N+1),-v(1:N)],-1:1,N,N)/hg^2;
            Theta_data=Pid*(Aab\Pid')/hg;
            Theta_sub=Pis*Theta_data*Pis';
            norm_data=u_data'*(Theta_data\u_data);
            norm_sub=u_sub'*(Theta_sub\u_sub);
            logdet=2*sum(log(diag(chol(Theta_data))));
            
            L_MLE(iter_a,iter_b)=norm_data+logdet;
            L_KF(iter_a,iter_b)=1-norm_sub/norm_data;
        end
    end
    
    [i,j]=find(L_MLE==min(min(L_MLE)));
    sol_MLE(iter,:)=[a_search(i(1)),b_search(j(1))];
%     v=kappa(t*hg,sol_MLE(iter,1),sol_MLE(iter,2))';
%     Aab=spdiags([-v(2:N+1),v(1:N)+v(2:N+1),-v(1:N)],-1:1,N,N)/hg^2;
%     testerr(iter,1)=norm(u-Aab\(Pid'*((Pid*(Aab\Pid'))\u_data)))*sqrt(hg);

    
    [i,j]=find(L_KF==min(min(L_KF)));
    sol_KF(iter,:)=[a_search(i(1)),b_search(j(1))];
%     v=kappa(t*hg,sol_KF(iter,1),sol_KF(iter,2))';
%     Aab=spdiags([-v(2:N+1),v(1:N)+v(2:N+1),-v(1:N)],-1:1,N,N)/hg^2;
%     testerr(iter,2)=norm(u-Aab\(Pid'*((Pid*(Aab\Pid'))\u_data)))*sqrt(hg);
    
    fprintf('No. %g, n=%g, MLE %g, %g, KF %g, %g, took %g s\n',iter,n,sol_MLE(iter,1),sol_MLE(iter,2),sol_KF(iter,1),sol_KF(iter,2),toc);
end

figure;
h=zeros(1,2);
h(1)=plot(a_search,L_MLE);
title('Loss function: Empirical Bayesian');
xlabel('\theta'); ylabel('loss');
[~,pos]=min(L_MLE);
hold on
h(2)=plot(a_search(pos),L_MLE(pos,end),'r*');
legend(h(2),'minimizer');

figure;
h2=zeros(1,2);
h2(1)=plot(a_search,L_KF);
title('Loss function: Kernel Flow');
xlabel('\theta'); ylabel('loss');
hold on
[~,pos]=min(L_KF);
h2(2)=plot(a_search(pos),L_KF(pos,end),'r*');
legend(h2(2),'minimizer');

% figure;
% subplot(1,2,1); semilogy(array_log2n,testerr(:,1)); title('test err: MLE');
% subplot(1,2,2); semilogy(array_log2n,testerr(:,2)); title('test err: KF');

function [y]=kappa(x,a,b)
    y=(x<=a).*ones(size(x))+(x>a).*b.*ones(size(x));
end

