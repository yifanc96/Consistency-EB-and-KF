figure
% cmap=rand(num_t,3);
load data_cmap cmap

axesfontsize=16;
axeslinewidth=1.8;
linelinewidth=1.8;
patchlinewidth=1.5;
set(0,'defaultaxesfontsize',axesfontsize,'defaultaxeslinewidth',axeslinewidth,...
    'defaultlinelinewidth',linelinewidth,'defaultpatchlinewidth',patchlinewidth)
for i=1:num_t
    semilogy(arr_q,L(i,:),'color',cmap(i,:));
    hold on
end
legend('t=0.5','t=0.75','t=1','t=1.25','t=1.5','t=1.75','t=2','t=2.25','t=2.5','t=2.75');
axis([4 9 10^(-10) 10^0]);
set(gca,'xtick',4:1:9);
set(gca,'ytick',10.^(-10:2:0));
set(gca,'yticklabel',{'10^{-10}','10^{-8}','10^{-6}','10^{-4}','10^{-2}','10^{0}'});
xlabel('q');
ylabel('L^2 error: averaged over the GP');
h=gcf;
myprint('L2_average_over_GP_for_KF_demonstration',h)
