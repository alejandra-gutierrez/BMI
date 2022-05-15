noTrials = 80; % number of trials to include - section for training data 
binWidth = 5; 
axis = 1;
[all_nrns, pos_time_axis] = trial_averaged_neurons(noTrials, binWidth); % create averages across all trials of each neuron in each direction 

all_psns = trial_averaged_hand_pos(axis, noTrials, binWidth); 

load('monkeydata_training.mat')
testTrial = 100;


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

%% ARCHIVE 
%% Linear Filter: position - training on full length of data
metric = 'p';  % 'p' for position, 'v' for velocity, 'a' for acceleration
test_input = cell(8,1);
test_output = cell(8,1);
training_input = all_nrns;
training_output = all_psns;
for direction = [1:1:8]
    test_input{direction} = trial(testTrial,direction).spikes();
    test_output{direction} = trial(testTrial,direction).handPos(1,:);
end 
plotLinearFilterPrediction(training_input, training_output, test_input, test_output, metric);
