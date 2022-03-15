noTrials = 80; % number of trials to include - section for training data 
binWidth = 5; 
[all_nrns, pos_time_axis] = trial_averaged_neurons(noTrials, binWidth); % create averages across all trials of each neuron in each direction 
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


%% Linear Regressions - X Position
axis = 1; % X direction movement 
direction = 1;
pred_x_pos = zeros(1, length(test_data{1, direction}));
% position 
all_psns = trial_averaged_hand_pos(axis, noTrials, binWidth); % average hand position taken over all trials
x_pos_weights = linearRegression(direction, all_nrns, all_psns); % Perform Linear Regression to get weights for predicting x-position
u = [];
p = 0;
for i = [1:1:length(trial(testTrial,direction).spikes(1,:))]
    for k = [1:1:98]
        p = p + x_pos_weights(k)*trial(testTrial,direction).spikes(k,i);
        u(:, k) = trial(testTrial,direction).spikes(k,:);
    end
    pred_x_pos(i) = p - transpose(x_pos_weights)*transpose(u(i,:));
end

act_x_pos = trial(testTrial,direction).handPos(axis, :);
figure
plot(pred_x_pos)
hold on 
plot(act_x_pos)
xlabel("Time (ms)")
ylabel("X-Position")
title("Direction k = " + direction)
legend(['Predicted: Linear Regression'], [("Actual Position: Trial " + testTrial)])
%% Linear Regression - X Velocity 

x_vel_weights = linearRegression(direction, all_nrns, all_vels); % Perform Linear Regression to get weights for predicting X-velocity 

u = [];
p = 0;
pred_x_vel = [];
for i = [1:1:length(trial(100,direction).spikes(1,:))]  % loop through all time points
    for k = [1:1:98]
        p = p + x_vel_weights(k)*trial(100,direction).spikes(k,i); % summation part of LR formula
        u(:, k) = trial(100,direction).spikes(k,:); % u vector from LR formula 
    end
    pred_x_vel(i) = p - transpose(x_vel_weights)*transpose(u(i,:));
end
act_x_vel = [0, diff(trial(testTrial,direction).handPos(axis,:))./diff([1:1:length(trial(testTrial,direction).handPos(axis,:))])];

figure
plot(pred_x_vel)
hold on 
plot(act_x_vel)
xlabel("Time (ms)")
ylabel("X Velocity")
title("Direction k = " + direction)
legend(['Predicted: Linear Regression'], [("Actual Position: Trial " + testTrial)])



%% Linear Regression - X Acceleration

x_acc_weights = linearRegression(direction, all_nrns, all_accs);
u = [];
p = 0;
pred_x_acc = [];
for i = [1:1:length(trial(100,direction).spikes(1,:))]  % loop through all time points
    for k = [1:1:98]
        p = p + x_acc_weights(k)*trial(100,direction).spikes(k,i); % summation part of LR formula
        u(:, k) = trial(100,direction).spikes(k,:); % u vector from LR formula 
    end
    pred_x_acc(i) = p - transpose(x_acc_weights)*transpose(u(i,:));
end
act_x_acc = [0, diff(act_x_vel)./diff([1:1:length(act_x_vel)])];

figure
plot(pred_x_acc)
hold on 
plot(act_x_acc)
xlabel("Time (ms)")
ylabel("X Acceleration")
title("Direction k = " + direction)
legend(['Predicted: Linear Regression'], [("Actual Position: Trial " + testTrial)])
