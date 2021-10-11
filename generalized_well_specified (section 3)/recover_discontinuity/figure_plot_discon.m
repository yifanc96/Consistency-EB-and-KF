load data_dis_cont_histogram sol_MLE sol_KF
axesfontsize=16;
axeslinewidth=1.8;
linelinewidth=1.8;
patchlinewidth=1.5;
set(0,'defaultaxesfontsize',axesfontsize,'defaultaxeslinewidth',axeslinewidth,...
    'defaultlinelinewidth',linelinewidth,'defaultpatchlinewidth',patchlinewidth)

figure
hist_ax=0.3:0.05:0.7;
temp=hist(sol_MLE(:,1),hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',0.3:0.05:0.7);
xlabel('\theta^{EB}');
h=gcf;
myprint('recover_discontinuity_well_specify_EB',h)

figure
hist_ax=0.3:0.05:0.7;
temp=hist(sol_MLE(:,1),hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',0.3:0.05:0.7);
xlabel('\theta^{KF}');
h=gcf;
myprint('recover_discontinuity_well_specify_KF',h)
