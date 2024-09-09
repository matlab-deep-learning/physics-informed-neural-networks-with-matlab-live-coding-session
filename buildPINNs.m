%% Load data
load massSpringDamperData.mat

xsolFcn = @(t)real(A.*exp(omega1.*t) + B.*exp(omega2.*t));

plotMassSpringDamperData(t0, tmax, tdata, xdata, tpinns, xsolFcn)

%% Build neural network
inputSize = 1;
outputSize = 1;
numHiddenUnits = 128;
layers = [ featureInputLayer(1) 
    fullyConnectedLayer(numHiddenUnits) 
    tanhLayer() 
    fullyConnectedLayer(numHiddenUnits) 
    tanhLayer() 
    fullyConnectedLayer(outputSize) ];
net = dlnetwork(layers);

deepNetworkDesigner(net)

%% Train the neural network

% Specify training hyperparameters.
numIterations = 5e3;

% Specify ADAM hyperparameters.
learnRate = 0.01;
mp = [];
vp = [];

% Prepare data for training.
tdata = dlarray(tdata, 'CB');
xdata = dlarray(xdata, 'CB');
tpinns = dlarray(tpinns, 'CB');

% Create training progress plot.
monitor = trainingProgressMonitor(Metrics=["Loss", "LossPINN", "LossData"]);
fig = figure();

% Accelerate model loss.
accFcn = dlaccelerate(@modelLoss);

for iteration = 1:numIterations
    [loss, gradients, lossPinn, lossData] = dlfeval(accFcn, net, tdata, xdata, tpinns, m, mu, k);

    [net, mp, vp] = adamupdate(net, gradients, mp, vp, iteration, learnRate);

    recordMetrics(monitor, iteration, ...
        Loss=loss, ...
        LossPINN=lossPinn, ...
        LossData=lossData);

    if mod(iteration, 50) == 0
        ttest = sort(rand(100,1)).*tmax;
        xtest = xsolFcn(ttest);
        xpred = predict(net, ttest);
        plotModelPredictions(fig, ttest, xtest, xpred, iteration);
    end
end

function [loss, gradients, lossPinn, lossData] = modelLoss(net, tdata, xdata, tpinns, m, mu, k)
lossPinn = pinnsLoss(net, tpinns, m, mu, k);

lossData = dataLoss(net, tdata, xdata);

loss = 0.1.*lossPinn + 0.05.*lossData;

gradients = dlgradient(loss, net.Learnables);
end

function loss = pinnsLoss(net, t, m, mu, k)
x = forward(net, t);

xt = dlgradient(sum(x,'all'), t, EnableHigherDerivatives=true);
xtt = dlgradient(sum(xt,'all'), t);

residual = m.*xtt + mu.*xt + k.*x;

loss = mean( residual.^2, 'all' );
end

function loss = dataLoss(net, t, xtarget)
x = forward(net, t);
loss = l2loss(x, xtarget);
end