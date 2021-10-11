clear
N=2^10; hg=1/(N+1); %fine mesh N
t=(0:1:N)+0.5; v=kappa(t*hg)'; 
A=spdiags([-v(2:N+1),v(1:N)+v(2:N+1),-v(1:N)],-1:1,N,N)/hg^2; clear v

[eigvec,eigval]=eig(full(A)); %eigen-pairs of A
alpha=2.5;  %ground truth
fprintf('ground truth for alpha is %g\n',alpha);
lambda=(diag(eigval).^alpha);
xi=randn(N,1); u=eigvec*(xi./sqrt(lambda)); % generate samples of GP

alpha_search=2:0.01:3; la=length(alpha_search); %grid search of alpha
array_log2n=6:6; ln=length(array_log2n); %n: num of data we observe; want to see the asymptotics when n goes to infinity
L_KF=zeros(la,ln); L_rL2=zeros(la,ln);  %loss function

sol_KF=zeros(ln,1);  % store estimators
sol_rL2=zeros(ln,1);

tic
normu=norm(u);
for iter=1:ln
    n=2^array_log2n(iter);  %num of data observed
    index=1:N/n:N; indexs=1:2:n; %index of the observed data and subsampling
    Pid=sparse(1:n,index,ones(n,1),n,N);  %Pid is for the data
    Pis=sparse(1:n/2,indexs,ones(n/2,1),n/2,n); %Pis is for the subsampling
    u_data=Pid*u; u_sub=Pis*u_data;

    for iter_a=1:la
        a=alpha_search(iter_a);
        Theta_global=eigvec*diag(1./diag(eigval.^a))*eigvec';
        Theta_data=Pid*Theta_global*Pid';
        Theta_sub=Pis*Theta_data*Pis';
        temp=Theta_data\u_data;
        u_interp=Theta_global*Pid'*(temp);
        norm_data=u_data'*(temp);
        norm_sub=u_sub'*(Theta_sub\u_sub);

        L_KF(iter_a,iter)=1-norm_sub/norm_data;
        L_rL2(iter_a,iter)=norm(u-u_interp)/normu;
    end
    
    
    [i]=find(L_KF(:,iter)==min(L_KF(:,iter)));
    sol_KF(iter)=alpha_search(i(1));
    
    [i]=find(L_rL2(:,iter)==min(L_rL2(:,iter)));
    sol_rL2(iter)=alpha_search(i(1));
    
    fprintf('No. %g, data n=%g, KF %g, rL2 %g, took %g s\n',iter,n,sol_KF(iter), sol_rL2(iter), toc);
end

figure
semilogy(alpha_search, L_rL2(:,end))


function [y]=kappa(x)
    y=ones(size(x));
end

% function [y]=kappa(x)
%     k=100;
%     W1=rand(k,1)-0.5; W2=rand(k,1)-0.5;
%     tmp_cos=cos((1:k)'*x);
%     tmp_sin=sin((1:k)'*x);
%     y=1+0.5*sin(W1'*tmp_cos+W2'*tmp_sin);%row vector
% end
