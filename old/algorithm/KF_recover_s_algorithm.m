% N: discrete points; n: data points; m: subsampled points
clear;

N=2^9; hg=1/(N+1); %fine mesh N
t=(0:1:N)+0.5; v=kappa(t*hg)'; 
A=spdiags([-v(2:N+1),v(1:N)+v(2:N+1),-v(1:N)],-1:1,N,N)/hg^2; clear v

[eigvec,eigval]=eig(full(A)); %eigen-pairs of A
s=2.5;  %ground truth
s_KF=(s-1/2)/2;
fprintf('ground truth for alpha is %g\n',s);
lambda=(diag(eigval).^s);
xi=randn(N,1); u=eigvec*(xi./sqrt(lambda)); % generate samples of GP

%-----measurements,data-----------
n=floor(N/2);
index=randsample(N,n); % random data
% index=1:8:N;
Pi=sparse(1:n,index,1,n,N);
u_data=Pi*u;

arr_rho=[];arr_L2=[];arr_testerr=[];

arr_t=2;
step=1;

while(1)
   t_now=arr_t(step);
   G_now=eigvec*diag(1./diag(eigval.^t_now))*eigvec';   
   
   m1=n;
%     m1=2;
   index1=randsample(n,m1); %random m1
%    index1=1:n;
   Pi1=sparse(1:m1,index1,1,m1,n); 
   ud=Pi1*u_data;
    
    m2=n/2;
%     m2=1;
    index2= randsample(m1,m2); %random Pi2
    %     index2=1:2:n;
    Pi2=sparse(1:m2,index2,1,m2,m1);  
    usub = Pi2 * ud;
    
    Pi1=Pi1*Pi;
    %objective calculation
    Pit=Pi2*Pi1;
    temp0 = G_now * Pit';
    temp = (Pit * temp0) \ usub;
    temp1 = usub' * temp;
%     u_re1=temp0*temp;
    u_re1=Pit'*temp;
    
    % note: gradient is of different form when A / G is used. (take care of 
    % formula for u_re1 and u_re2
    
    tempp0 = G_now * Pi1';
    tempp = (Pi1 * tempp0)\ud;
    tempp1 = ud' * tempp;
%     u_re2 = tempp0*tempp;
    u_re2 = Pi1'*tempp;
    
    arr_rho=[arr_rho,1-temp1/tempp1];
    arr_L2=[arr_L2,norm(u_re1-u_re2)^2/norm(u_re2)^2];
    arr_testerr=[arr_testerr,norm(u-tempp0*tempp)^2];
     
    Bh=eigvec*diag((1./diag(eigval.^t_now)).*log(diag(eigval)))*eigvec'; 
    
    grad_t=((1-arr_rho(step))*u_re2'*Bh*u_re2-u_re1'*Bh*u_re1)/(tempp1);
   
    if step>=2000
        disp(step);
        break
    end
    
    if norm(grad_t)<=1e-6
        break
    end
    step_size=1e-3;
%     t_next=t_now-step_size*grad_t/norm(grad_t);
    t_next=t_now-step_size*grad_t;
    fprintf('step : %d, rho : %g, t now: %g, test err is %g\n',step,arr_rho(step),t_now,arr_testerr(step));
%     fprintf('step : %d, rho : %g, t now: %g\n',step,arr_rho(step),t_now);
    
    step=step+1;
    arr_t=[arr_t,t_next];
end



function [y]=kappa(x)
    y=ones(size(x));
end







