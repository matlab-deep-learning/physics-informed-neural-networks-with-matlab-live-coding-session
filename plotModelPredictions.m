function plotModelPredictions(fig, ttest, xtest, xpred, iteration)
figure(fig);
fig.Visible = true;

plot(ttest,xtest,'b-',DisplayName='Exact solution',LineWidth=2.5);
hold on
plot(ttest,xpred,'--',DisplayName='Model prediction',LineWidth=2.5,Color="r");

xlim([0 10]); ylim([min(xtest) 2])

title(sprintf('Iteration %d',iteration));
legend('Location','NorthEast');
hold off

ylabel('x')
xlabel('t')
ax = gca;
ax.FontSize = 16;
ax.LineWidth = 1.5;
end