load data_mis_dis_cont a_search L_MLE L_KF
axesfontsize=16;
axeslinewidth=1.8;
linelinewidth=1.8;
patchlinewidth=1.5;
set(0,'defaultaxesfontsize',axesfontsize,'defaultaxeslinewidth',axeslinewidth,...
    'defaultlinelinewidth',linelinewidth,'defaultpatchlinewidth',patchlinewidth)

figure
h=zeros(1,2);
h(1)=plot(a_search,L_MLE(:,end));
tmp=L_MLE(:,end);
i=find(tmp==min(tmp));
title('Loss function: Empirical Bayesian');
xlabel('\theta'); ylabel('loss');
hold on
h(2)=plot(a_search(i(1)),L_MLE(i(1),end),'r*');
legend(h(2),'minimizer');
h=gcf;
myprint('mis_specification_loss_EB',h)

figure;
h2=zeros(1,2);
h2(1)=semilogy(a_search,L_KF(:,end));
tmp=L_KF(:,end);
i2=find(tmp==min(tmp));
title('Loss function: Kernel Flow');
xlabel('\theta'); ylabel('loss');
hold on
h2(2)=plot(a_search(i2(1)),L_KF(i2(1),end),'r*');
legend(h2(2),'minimizer');
h=gcf;
myprint('mis_specification_loss_KF',h)

