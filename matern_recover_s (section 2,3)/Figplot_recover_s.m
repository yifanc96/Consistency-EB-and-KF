load data_recover_s

axesfontsize=16;
axeslinewidth=1.8;
linelinewidth=1.8;
patchlinewidth=1.5;
set(0,'defaultaxesfontsize',axesfontsize,'defaultaxeslinewidth',axeslinewidth,...
    'defaultlinelinewidth',linelinewidth,'defaultpatchlinewidth',patchlinewidth)
% figure: EB_loss_1d
figure
h=zeros(1,2);
h(1)=plot(s_search,L_MLE(:,end));
title('Loss function: Empirical Bayesian');
xlabel('t'); ylabel('loss');
hold on
i=find(L_MLE==min(L_MLE));
h(2)=plot(s_search(i),L_MLE(i,end),'r*');
legend(h(2),'minimizer');
h=gcf;
myprint('EB_loss_1d',h)

% figure: KF_loss_1d
figure;
h2=zeros(1,2);
h2(1)=semilogy(s_search,L_KF(:,end));
title('Loss function: Kernel Flow');
xlabel('t'); ylabel('loss');
hold on
i=find(L_KF==min(L_KF));
h2(2)=plot(s_search(i),L_KF(i,end),'r*');
legend(h2(2),'minimizer');
h=gcf;
myprint('KF_loss_1d',h)