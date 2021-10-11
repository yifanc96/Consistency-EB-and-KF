clear
N=2^9; hg=1/(N+1);
t=(0:1:N)+0.5; v=kappa(t*hg)'; 
A=spdiags([-v(2:N+1),v(1:N)+v(2:N+1),-v(1:N)],-1:1,N,N)/hg^2; clear v

[eigvec,eigval]=eig(full(A));
logdet=-2*sum(log(diag(chol(A))));
m=floor((N+1)/2);
Nm=1:2:N;   %deterministic subsampling
Pi=sparse(1:m,Nm,ones(m,1),m,N); 
Nm2=2:2:N;
Pi2=sparse(1:m,Nm2,ones(m,1),m,N);

alpha=2.5; gamma=exp(1.5); 
xi=randn(N,1);
lambda=gamma*(diag(eigval).^alpha);
u=eigvec*(xi./sqrt(lambda));
us=Pi*u;
us2=Pi2*u;

alpha_search=0.5:0.01:3; la=length(alpha_search);
gamma_search=1:0.2:2; lg=length(gamma_search);

L_MLE=zeros(la,lg);
L_HMLE=zeros(la,lg); L_HMLE2=zeros(la,lg); L_HMLE_ave=zeros(la,lg);
L_KF=zeros(la,lg);

for iter_a=1:la
    for iter_g=1:lg
        a=alpha_search(iter_a);
        g=exp(gamma_search(iter_g));
        Thetah1=Pi*eigvec*diag(1./diag(g*eigval.^a))*eigvec'*Pi';
        Thetah2=Pi2*eigvec*diag(1./diag(g*eigval.^a))*eigvec'*Pi2';
        norm1=us'*(Thetah1\us); norm1_2=us2'*(Thetah2\us2);
        norm2=u'*eigvec*(g*eigval.^a)*eigvec'*u;
        logdet1=2*sum(log(diag(chol(Thetah1))));
        logdet2=2*sum(log(diag(chol(Thetah2))));
        
        L_MLE(iter_a,iter_g)=norm2-N*log(g)+a*logdet;
        L_HMLE(iter_a,iter_g)=norm2-norm1-N*log(g)+a*logdet-logdet1;
        L_HMLE2(iter_a,iter_g)=norm2-norm1_2-N*log(g)+a*logdet-logdet2;
        L_HMLE_ave(iter_a,iter_g)=1/2*(L_HMLE(iter_a,iter_g)+L_HMLE2(iter_a,iter_g))/2;
        L_KF(iter_a,iter_g)=1-norm1/norm2;
    end
end

figure;
subplot(2,3,1);
contourf(alpha_search,gamma_search,L_MLE'); xlabel('\alpha'); ylabel('\gamma'); title('MLE');
[i,j]=find(L_MLE==min(min(L_MLE)));
hold on
plot(alpha_search(i),gamma_search(j),'p');

% figure;
subplot(2,3,2);
contourf(alpha_search,gamma_search,L_HMLE'); xlabel('\alpha'); ylabel('\gamma'); title('HMLE:one half');
[i,j]=find(L_HMLE==min(min(L_HMLE)));
hold on
plot(alpha_search(i),gamma_search(j),'p');

% figure;
subplot(2,3,3);
contourf(alpha_search,gamma_search,L_HMLE2'); xlabel('\alpha'); ylabel('\gamma'); title('HMLE:another half');
[i,j]=find(L_HMLE2==min(min(L_HMLE2)));
hold on
plot(alpha_search(i),gamma_search(j),'p');

% figure;
subplot(2,3,4);
contourf(alpha_search,gamma_search,L_HMLE_ave'); xlabel('\alpha'); ylabel('\gamma'); title('HMLE:average');
[i,j]=find(L_HMLE_ave==min(min(L_HMLE_ave)));
hold on
plot(alpha_search(i),gamma_search(j),'p');

% figure;
subplot(2,3,5);
contourf(alpha_search,gamma_search,log(L_KF')); xlabel('\alpha'); ylabel('\gamma'); title('KF');
[i,j]=find(L_KF==min(min(L_KF)));
hold on
plot(alpha_search(i),gamma_search(j),'p');


function [y]=kappa(x)
    y=ones(size(x));
end

