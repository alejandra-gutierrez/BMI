noTrials = 80; % number of trials to include - section for training data 
binWidth = 5; 
axis = 1;
[all_nrns, pos_time_axis] = trial_averaged_neurons(noTrials, binWidth); % create averages across all trials of each neuron in each direction 

all_psns = trial_averaged_hand_pos(axis, noTrials, binWidth); 

load('monkeydata_training.mat')
testTrial = 100;
%% Linear Filter: position 
xTrain = [];
sTrain = [];
f = [];
sTest = [];
xActual = [];
xFit = [];

direction = 1;
numBin = length(all_psns{direction}); % number of bins
xTrain = all_psns{direction}; % training data: position outputs
for neuron = [1:1:98] % training data: spike inputs - get the inputs in the required format for training 
    sTrain(:,neuron) = all_nrns{neuron, direction};
end
% sTrain is T x 98 - 98 neurons/columns, T time chunks/rows
sTrain = [sTrain ones(length(sTrain),1)]; % adding vector of ones acts to allow for a 'baseline' firing rate of the neurons
f = inv(sTrain'*sTrain)*sTrain'*xTrain'; % creates linear filter (effectively finds weights for linear regression)


sTest = [trial(testTrial,direction).spikes(); ones(1,length(trial(testTrial,direction).spikes()))]; % test data: spikes for trial 100 as test trial
xActual = trial(testTrial,direction).handPos(1,:); % actual position data for trial 100 
xFit = sTest'*f; % apply filter to predict x position from test input data

 
figure 
plot(xFit) 
hold on 
plot(xActual)
xlabel('Time (ms)')
ylabel('Position (mm)')
title("Direction k = " + direction)
legend(['Predicted: Linear'], ['Actual'])

%% Linear Filter: velocity
xTrain = [];
sTrain = [];
f = [];
sTest = [];
xActual = [];
xFit = [];

all_vels = cell(1,8); % initialise cell array for velocity
vel_time_axis = cell(1, 8); % initialise cell array for velocity time points

% calculate hand velocity over time 
% time points for velocity taken as midpoint of time interval over which velocity calculated
for direction = [1:1:8]
    all_vels{direction} = [0, diff(all_psns{direction})./diff(pos_time_axis{direction})];
    vel_time_axis{direction} = [1, (pos_time_axis{direction}(1:end-1) + pos_time_axis{direction}(2:end))/2];
end

figure
for direction = [1:1:8]
    xTrain = [];
    sTrain = [];
    f = [];
    sTest = [];
    xActual = [];
    xFit = [];
    numBin = length(all_vels{direction}); % number of bins
    xTrain = all_vels{direction}; % training data: velocity outputs
    for neuron = [1:1:98] % training data: spike inputs - get the inputs in the required format for training 
        sTrain(:,neuron) = all_nrns{neuron, direction};
    end
    % sTrain is T x 98 - 98 neurons/columns, T time chunks/rows
    sTrain = [sTrain ones(length(sTrain),1)]; % adding vector of ones acts to allow for a 'baseline' firing rate of the neurons

    % remove linearly dependent columns - if identical columns are present,
    % only one is kept 
    M = sTrain;
    %https://www.mathworks.com/matlabcentral/answers/574543-algorithm-to-extract-linearly-dependent-columns-in-a-matrix#answer_474601
    [Q,R,p] = qr(M,'vector');
    dr = abs(diag(R));
    if dr(1)
        tol = 1e-10;
        r = find(dr>=tol*dr(1),1,'last');
        ci = p(1:r); % here is the index of independent columns
        di = setdiff([1:1:98],ci);
    else
        r = 0;
        ci = [];

    end
    ci = sort(ci); % sort indicies of linearly independent columns
    di = sort(di); % sort indices of linearly dependent columns

    % Submatrix with r columns (and full column rank).
    Mind=M(:,ci)
    Dind = M(:,di)
    % Those three rank estimation should be equals 
    % if it's not then the cause if MATLAB selection of tolerance for rank differs with the above
    % and usage of more robust SVD algorithm for rank estimation
    rank(Mind)
    rank(M)
    r

    % sTrain = Mind; % training data now becomes only linearly independent neuron data columns
    sTrain_dep = Dind;
    % set test data to only be for linearly independent neurons 
    j = 1;
    if max(ci) > 98
        for i = [1:1:length(ci)-1]
            sTest(j,:) = trial(testTrial,direction).spikes(ci(j),:);
            j = j + 1;
        end
        sTest(j,:) = ones(1,length(trial(testTrial,direction).spikes()));
    end
    f = inv(Mind'*Mind)*Mind'*xTrain'; % creates linear filter (effectively finds weights for linear regression)


    % sTest = [trial(testTrial,direction).spikes(); ones(1,length(trial(testTrial,direction).spikes()))]; % test data: spikes for trial 100 as test trial
    xActual = [0, diff(trial(testTrial, direction).handPos(1,:))./diff([1:1:length(trial(testTrial, direction).handPos(1,:))])]; % actual velocity data for trial 100 
    xFit = sTest'*f;  % apply filter to predict x velocity from test input data


    subplot(4, 2, direction)
    plot(xFit) 
    hold on 
    plot(xActual)
    xlabel('Time (ms)')
    ylabel('Velocity (m/s)')
    title("Direction k = " + direction)
    legend(['Predicted: Linear'], ['Actual'])
end

%% MAP Estimation 
xCenter = [-99.5:1:99.5];
numNeurons = 98;
for n = 1:1:numNeurons
    coeff(n,:) = glmfit(all_psns{direction}(1:end-floor(100/binWidth)), sTrain(:,n), 'poisson');
    sFit(n,:) = exp(coeff(n,1) + coeff(n,2)*xCenter);
end
%% 
for t = 1:length(trial(testTrial,direction).spikes(1,1:end-100))
    frTemp = trial(testTrial,direction).spikes(1:numNeurons,t);
    prob = poisspdf(repmat(frTemp,1,200),sFit);
    probSgivenX(t,:) = prod(prob);
end
%% 
probX = histc(xTrain,[-100:1:100]);
probX = probX(1:200)/sum(probX(1:200));
for t = 1:length(trial(testTrial,direction).spikes(1,1:end-100))
    probXgivenS(t,:) = probSgivenX(t,:).*probX;
    [temp maxInd] = max(probXgivenS(t,:));
    mapS(t) = xCenter(maxInd);
end

figure 
plot(mapS) 
hold on 
plot(xActual)
xlabel('Time (ms)')
ylabel('Position (mm)')
title('Direction k = 1')
legend(['Predicted: MAP'], ['Actual'])

%%
M = 2; % number of hidden layers 
N = 98; % Number of responses (responding things?) at time point to give output 

noTrials = 80; % number of trials to include - section for training data 
binWidth = 5; 
[all_nrns, pos_time_axis] = trial_averaged_neurons(noTrials, binWidth); % create averages across all trials of each neuron in each direction 

[training_data, t_ax] = collateTrainData(noTrials, binWidth, trial);

[test_data,~] = trial_averaged_neurons(noTrials, binWidth);

axis = 1; % X direction movement 
% axis = 2; % Y direction movement

all_psns = trial_averaged_hand_pos(axis, noTrials, binWidth); 



    % average hand position taken over all trials

all_vels = cell(1,8); % initialise cell array for velocity
vel_time_axis = cell(1, 8); % initialise cell array for velocity time points

% calculate hand velocity over time 
% time points for velocity taken as midpoint of time interval over which
% velocity calculated
for direction = [1:1:8]
    all_vels{direction} = [0, diff(all_psns{direction})./diff(pos_time_axis{direction})];
    vel_time_axis{direction} = [1, (pos_time_axis{direction}(1:end-1) + pos_time_axis{direction}(2:end))/2];
end

% acceleration 
all_accs = cell(1, 8);
acc_time_axis = cell(1, 8);
for direction = [1:1:8]
    all_accs{direction} = [0, diff(all_vels{direction})./diff(vel_time_axis{direction})];
    acc_time_axis{direction} = [1, (vel_time_axis{direction}(1:end-1) + vel_time_axis{direction}(2:end))/2];
end

% Testing Regression: parameters
load('monkeydata_training.mat')
testTrial = 100; % Trial to use as 'actual' data to compare to predicted values 
%% X position
response_data = zeros(length(all_nrns{neuron, direction}),98); % predictors - 98 neurons, T time 'chunks' - T x 98 array 
for neuron = 1:1:98
    response_data(:,neuron) = all_nrns{neuron, direction};
end

% response_data contains spike data - T x 98 - each 1x98 row used for a single time chunk 
gprMdl = fitrgp(response_data, all_psns{direction});
predicted_pos = predict(gprMdl, transpose(trial(testTrial,direction).spikes()));

figure
plot(trial(testTrial, direction).handPos(1,:))
hold on 
% plot(predicted_x_position);
plot(predicted_pos);
legend(['Actual Hand Position'], ['Predicted Hand Position'])
%% X velocity
predicted_x_position = [];
predicted_vel = [];

movement_tuning_curve_data = cell(98,8);
for direction = [1:1:8]
    for neuron = [1:1:98]
        tuning_curve_temp = [];
        
        tuning_curve_temp(1,:) = all_vels{direction}(ceil(299/binWidth):end-floor(100/binWidth)); 
        tuning_curve_temp(2,:) = all_nrns{neuron, direction}(ceil(299/binWidth):end-floor(100/binWidth));
        [~, order] = sort(tuning_curve_temp(1, :)); 
        tuning_curve_temp = tuning_curve_temp(:, order);
       
        movement_tuning_curve_data{neuron, direction} = tuning_curve_temp(1:2,:); 
    end
end
response_data = zeros(length(movement_tuning_curve_data{neuron, direction}),98); % predictors - 98 neurons, T time 'chunks' - T x 98 array 
for neuron = 1:1:98
    response_data(:,neuron) = movement_tuning_curve_data{neuron, direction}(2,:);
end

gprMdl = fitrgp(response_data, movement_tuning_curve_data{neuron, direction}(1,:));
predicted_vel = predict(gprMdl, transpose(trial(testTrial,direction).spikes(:, 301:end-100)));

startHandPos_x = [trial(testTrial, direction).handPos(1,1)];
predicted_x_position(1) = startHandPos_x;
for t = [2:1:length(trial(testTrial, direction).handPos(1,301:end-100))]
    predicted_x_position(t) = predicted_x_position(t-1)+predicted_vel(t);
end
figure
plot(trial(testTrial, direction).handPos(1,300:end-100))
hold on 
plot(predicted_x_position);
plot(predicted_vel);
legend(['Actual Hand Position'], ['Predicted Hand Position'], ['Predicted Velocity'])
%% X velocity
gprMdl = fitrgp(response_data, all_vels{direction});
predicted_vel = predict(gprMdl, transpose(trial(testTrial,direction).spikes()));

startHandPos_x = [trial(testTrial, direction).handPos(1,1)];
predicted_x_position(1) = startHandPos_x;
for t = [2:1:length(trial(testTrial, direction).handPos(1,:))]
    predicted_x_position(t) = predicted_x_position(t-1)+predicted_vel(t);
end
figure
plot(trial(testTrial, direction).handPos(1,:))
hold on 
plot(predicted_x_position);
plot(predicted_vel);
legend(['Actual Hand Position'], ['Predicted Hand Position'], ['Predicted Velocity'])