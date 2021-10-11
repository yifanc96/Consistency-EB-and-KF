
load data_variance_misspecify sol_KF sol_MLE  MLE_ave_err_s KF_ave_err_s
axesfontsize=16;
axeslinewidth=1.8;
linelinewidth=1.8;
patchlinewidth=1.5;
set(0,'defaultaxesfontsize',axesfontsize,'defaultaxeslinewidth',axeslinewidth,...
    'defaultlinelinewidth',linelinewidth,'defaultpatchlinewidth',patchlinewidth)

figure
hist_ax=2.3:0.05:2.7;
temp=hist(sol_MLE(:,1),hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',2.3:0.05:2.7);
xlabel('s^{EB}');
h=gcf;
myprint('misspecify_recover_regularity_EB',h)


figure
hist_ax=0.8:0.05:1.2;
temp=hist(sol_KF(:,1),hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',0.8:0.05:1.2);
xlabel('s^{KF}');
h=gcf;
myprint('misspecify_recover_regularity_KF',h)


