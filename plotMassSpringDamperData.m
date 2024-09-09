function plotMassSpringDamperData(t0, tmax, tdata, xdata, tpinns, xsolFcn)
tt = linspace(t0, tmax, 100);
xt = xsolFcn(tt);

figure()
hold on
plot(tt, xt, 'b', LineWidth=2.5, DisplayName='Exact solution')
legend(); 
ylabel('x')
xlabel('t')
ax = gca; 
ax.FontSize = 16; 
ax.LineWidth = 1.5;
hold on;
scatter(tdata, xdata, 48, [0.4660 0.6740 0.1880], ...
    DisplayName='Data loss points', ...
    LineWidth=2);
scatter(tpinns, zeros(length(tpinns),1), 30, ...
    [0.4660 0.6740 0.1880], ...
    "filled", ...
    DisplayName='PINNs loss points'); 
hold off
end