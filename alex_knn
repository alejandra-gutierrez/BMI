
clear;close all; clc;

%% Get the number of spikes per trial per direction
% we only use 80 trials so that the other 20 trials are used for validation
% the resulting variable 'rate' is of size (98, 640) => (98 neurons, 8 directions * 80 trials)
% this is only done for the first 300ms, as we want data before movement starts
% k = angles
% n = trials
% neuron = neurons


training_data = load('monkeydata_training.mat');
trial = training_data.trial;
training_data = trial(1:80, :);


test_point = sum(trial(81,7).spikes(: ,1:300), 2);

for k=1:8
    for n=1:80
        label((k-1)*80+n) = k;
        for neuron=1:98
            rate(neuron, n*k)=sum(training_data(n,k).spikes(neuron, 1:300));
        end
    end 
end 


%% Calculate eucledian distance between the number of spikes for each trials and directions
% the result is a matrix of 640 x 640, with diagonal 0.

for t1 =1:640
    for t2=1:640
        dist(t1, t2) = sqrt(sum((rate(:, t1) - rate(:, t2)).*(rate(:, t1) - rate(:, t2))));
    end 
end 


%% find the predicted angle by selecting the one that was most frequently in the neirest neighbours
% Store index of the neirest neighbour
% Then get the predicted label for that data point
n_neighbours = 10;

[sortedDistance, indexes] = sort(dist, 1);
nearest = indexes(2:n_neighbours+1, :);

for col=1:640
    nearest_angle(1:n_neighbours, col) = label(nearest(:, col));
    predicted_angle(col) = mode(nearest_angle(:, col));
end 


%% 
for col = 1:640
    difference(col) = label(:, col) - predicted_angle(:, col);
end 

figure
plot(sort(abs(difference)))


%%
test_point = sum(trial(88,5).spikes(: ,1:300), 2);

n_neighbours = 6;
for col=1:640

    d(col)= sqrt(sum((test_point(:) - rate(:, col)).^2));
end
[sorted_D, test_ind] = sort(d);
nearest = test_ind(2:n_neighbours)
nearest_angles = label(nearest)
predicted_angle = mode(nearest_angles)


%% adding logic to infer the label from the angle (pi), not the 1:8
% make a list of unit direction vectors for each angle (cartesian, x, y)
k_list = 1:8;
theta = (40*k_list-10)/180*pi;
unit_vect_list = [cos(theta); sin(theta)];

nearest_vects = unit_vect_list(nearest_angles);
predicted_vect = mean(nearest_vects, 1);

ang = atan(predicted_vect(2)/predicted_vect(1)); % that's an angle

if ang < 0
    ang = ang+2*pi
end 

k_deduced = (180*ang/pi + 10)/40

