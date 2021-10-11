clear
N=2^9; hg=1/(N+1);
a_true=0.5; b_true=2;
fprintf('ground truth for a,b is %g, %g\n',a_true,b_true);
t=(0:1:N)+0.5; v=kappa(t*hg,a_true,b_true)'; 
A=spdiags([-v(2:N+1),v(1:N)+v(2:N+1),-v(1:N)],-1:1,N,N)/hg^2; 


v=laplace_mtx(t*hg)'; 
A_lap=spdiags([-v(2:N+1),v(1:N)+v(2:N+1),-v(1:N)],-1:1,N,N)/hg^2;
[eigvec,eigval]=eig(full(A_lap));
s=1;
lambda=diag(eigval).^s; 

a_search=0.3:0.005:0.7; la=length(a_search);
b_search=b_true;
lb=length(b_search);
L_MLE=zeros(la,lb); L_KF=zeros(la,lb); %loss function
array_log2n=7; ln=length(array_log2n);

tic
alpha_postulate=1;
lap_postulate=eigvec*diag(diag(eigval).^alpha_postulate)*eigvec';
G_postulate=eigvec*diag(1./diag(eigval).^alpha_postulate)*eigvec';

tot_iter=1; % total instances
sol_MLE=zeros(tot_iter,2); sol_KF=sol_MLE; % store estimators

for iter_sample=1:tot_iter
    xi=randn(N,1); 
    u=A\(eigvec*(xi./sqrt(lambda))); %A is FD stiffness mtx;
    for iter=1:ln
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
                
                tmp=Aab\Pid';
                Theta_data=tmp'*G_postulate*tmp;
                Theta_sub=Pis*Theta_data*Pis';
                norm_data=u_data'*(Theta_data\u_data);
                norm_sub=u_sub'*(Theta_sub\u_sub);
                logdet=2*sum(log(diag(chol(Theta_data/hg))));
                
                L_MLE(iter_a,iter_b)=norm_data+logdet;
                L_KF(iter_a,iter_b)=1-norm_sub/norm_data;
            end
        end
        
        [i,j]=find(L_MLE==min(min(L_MLE)));
        sol_MLE(iter_sample,:)=[a_search(i(1)),b_search(j(1))];
        
        [i2,j2]=find(L_KF==min(min(L_KF)));
        sol_KF(iter_sample,:)=[a_search(i2(1)),b_search(j2(1))];
        
        fprintf('No. %g, n=%g, MLE %g, %g, KF %g, %g, took %g s\n',iter,n,sol_MLE(iter,1),sol_MLE(iter,2),sol_KF(iter,1),sol_KF(iter,2),toc);
    end
end

%save data
% save data_dis_cont_histogram sol_MLE sol_KF

% figure
% hist_ax=0.3:0.04:0.7;
% temp=hist(sol_MLE(:,1),hist_ax);
% bar(hist_ax,temp)
% set(gca,'XTick',0.3:0.04:0.7);
% xlabel('\theta^{EB}');
% 
% figure
% hist_ax=0.3:0.04:0.7;
% temp=hist(sol_KF(:,1),hist_ax);
% bar(hist_ax,temp)
% set(gca,'XTick',0.3:0.04:0.7);
% xlabel('\theta^{KF}');

axesfontsize=16;
axeslinewidth=1.8;
linelinewidth=1.8;
patchlinewidth=1.5;
set(0,'defaultaxesfontsize',axesfontsize,'defaultaxeslinewidth',axeslinewidth,...
    'defaultlinelinewidth',linelinewidth,'defaultpatchlinewidth',patchlinewidth)

figure
h=zeros(1,2);
h(1)=plot(a_search,L_MLE(:,end));
title('Loss function: Empirical Bayesian');
xlabel('\theta'); ylabel('loss');
hold on
h(2)=plot(a_search(i(1)),L_MLE(i(1),end),'r*');
legend(h(2),'minimizer');
% h=gcf;
% myprint('recover_discontinuity_well_specify_loss_EB',h)

figure;
h2=zeros(1,2);
h2(1)=semilogy(a_search,L_KF(:,end));
title('Loss function: Kernel Flow');
xlabel('\theta'); ylabel('loss');
hold on
h2(2)=plot(a_search(i2(1)),L_KF(i2(1),end),'r*');
% legend(h2(2),'minimizer');h=gcf;
% myprint('recover_discontinuity_well_specify_loss_KF',h)


% save dis_cont_loss_func a_search L_MLE L_KF


function [y]=kappa(x,a,b)
    y=(x<a).*ones(size(x))+(x>=a).*b.*ones(size(x));
end


function [y]=laplace_mtx(x)
    y=ones(size(x));
end
