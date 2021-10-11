
axesfontsize=16;
axeslinewidth=1.8;
linelinewidth=1.8;
patchlinewidth=1.5;
set(0,'defaultaxesfontsize',axesfontsize,'defaultaxeslinewidth',axeslinewidth,...
    'defaultlinelinewidth',linelinewidth,'defaultpatchlinewidth',patchlinewidth)

%% EB, KF recover sigma and s
load data_matern_MLE_s_sigma sol_MLE_s sol_MLE_sigma
h=figure;
hist_ax=2:0.1:3;
temp=hist(sol_MLE_s,hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',2:0.1:3 );
xlabel('s^{EB}');
myprint('s_sigma_together_for_s',h)

h=figure;
hist_ax=-0.5:0.1:0.5;
temp=hist(sol_MLE_sigma,hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',-0.5:0.1:0.5 );
xlabel('log \sigma^{EB}');
myprint('s_sigma_together_for_sigma',h)

%% EB, KF recover s and tau

load data_matern_MLE_s_tau sol_MLE_s sol_MLE_tau sol_KF_s sol_KF_tau
h=figure;
hist_ax=2:0.1:3;
temp=hist(sol_MLE_s,hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',2:0.1:3 );
xlabel('s^{EB}');

myprint('s_tau_recovery_MLE_hist_s',h);

h=figure;
hist_ax=-2:0.4:2;
temp=hist(sol_MLE_tau,hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',-2:0.4:2 );
xlabel('log \tau^{EB}');
myprint('s_tau_recovery_MLE_hist_tau',h);

h=figure;
hist_ax=0.7:0.05:1.3;
temp=hist(sol_KF_s,hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',0.7:0.1:1.3 );
xlabel('s^{KF}');
myprint('s_tau_recovery_KF_hist_s',h);

h=figure;
hist_ax=-2:0.4:2;
temp=hist(sol_KF_tau,hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',-2:0.4:2 );
xlabel('log \tau^{KF}');
myprint('s_tau_recovery_KF_hist_tau',h);


%% 
load data_matern_tau sol_MLE_tau sol_KF_tau sol_KF2_tau

h=figure;
hist_ax=-2:0.4:2;
temp=hist(sol_MLE_tau,hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',-2:0.4:2 );
xlabel('log \tau^{EB}');
myprint('tau_recovery_MLE',h);

figure;
hist_ax=-2:0.25:2;
temp=hist(sol_KF_tau,hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',-2:0.4:2 );
xlabel('log \tau^{KF}');
myprint('tau_recovery_KF',h);

figure;
hist_ax=-2:0.4:2;
temp=hist(sol_KF2_tau,hist_ax);
bar(hist_ax,temp)
set(gca,'XTick',-2:0.4:2 );
xlabel('log \tau^{KF}');
myprint('tau_recovery_KF2',h);

