load data_variable_variance sol_KF sol_MLE  MLE_ave_err_s KF_ave_err_s
axesfontsize=16;
axeslinewidth=1.8;
linelinewidth=1.8;
patchlinewidth=1.5;
set(0,'defaultaxesfontsize',axesfontsize,'defaultaxeslinewidth',axeslinewidth,...
    'defaultlinelinewidth',linelinewidth,'defaultpatchlinewidth',patchlinewidth)
figure
hist_ax=2.4:0.02:2.6;
temp=hist(sol_MLE(:,1),hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',2.4:0.04:2.6);
xlabel('s^{EB}');
h=gcf;
myprint('hist_EB_s_variable_coeff',h)

figure
hist_ax=0.85:0.05:1.2;
temp=hist(sol_KF(:,1),hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',0.85:0.05:1.2);
xlabel('s^{KF}');
h=gcf;
myprint('hist_KF_s_variable_coeff',h);
