% plot figure for recovering sigma (matern kernel)
axesfontsize=16;
axeslinewidth=1.8;
linelinewidth=1.8;
patchlinewidth=1.5;
set(0,'defaultaxesfontsize',axesfontsize,'defaultaxeslinewidth',axeslinewidth,...
    'defaultlinelinewidth',linelinewidth,'defaultpatchlinewidth',patchlinewidth)

load data_recover_sigma

% figure: recover_sigma
h=figure;
h(1)=plot(sigma_search,L_MLE);
i=find(L_MLE==min(L_MLE));
title('Loss function: Empirical Bayesian');
xlabel('log \varsigma'); ylabel('loss');
hold on
h(2)=plot(sigma_search(i),L_MLE(i),'r*');
legend(h(2),'minimizer');
h=gcf;

myprint('recover_sigma',h)
% print('-painters','-dpdf','-r0','recover_sigma');