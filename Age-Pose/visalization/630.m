plot(predictions(:,id(i)),'--ro','LineWidth',2,...
    'MarkerEdgeColor','r',...
    'MarkerFaceColor','r',...
    'MarkerSize',5)
grid on
axis([0 85 0 0.12])
ylabel('','color','blue','FontSize',12)
xlabel('Age','color','blue','FontSize',12)
set(gca,'YTick',0:0.01:0.12);
set(gca,'XTick',0:10:85);
%     text(50,0.07,'N(31,4.2357)','color','blue','HorizontalAlignment','right','FontSize',12)
set(gca,'FontSize',15)
% set(gcf,'Position',[680   49    1064   915])
% set(gcf,'Position',[325  2   1551  9625])
print(gcf,'-depsc',['figure/chalearn_ld', num2str(i),'.eps']);


