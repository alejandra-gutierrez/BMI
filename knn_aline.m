%% Test knn classification
% Aline's code
%clear; close all;
%% setup
data = load('monkeydata_training.mat');
trial = data.trial;
% Randomly select elements for training and testing
ix = randperm(length(trial));
training_data = trial(ix(1:50),:);
testing_data = trial(ix(51:end), :);

[N_trials_tr, N_angles] = size(training_data);
N_trials_test = size(testing_data,1);
N_neurons = size(trial(1).spikes, 1);

k_list = 1:N_angles;
theta = (40*k_list-10)/180*pi;
unit_vect_list = [cos(theta); sin(theta)];

%n_neighbours = 5;
t_pre_mvt = 320;

%% Get the spike rate
train_rates = zeros(N_angles*N_trials_tr, N_neurons);
labels_tr = zeros(1, N_angles*N_trials_tr);
for k_it = 1:N_angles
    for n_it = 1:N_trials_tr
        train_rates( (k_it-1)*N_trials_tr + n_it, :) = sum(training_data(n_it, k_it).spikes(:, 1:t_pre_mvt), 2)';           
        labels_tr( (k_it-1)*N_trials_tr + n_it) = k_it;
    end
end

knn0 = fitcknn(train_rates, labels_tr); % model from package

%% Find mutual distance between points in training set
[N_tr, M_tr] = size(train_rates); % [N_angles*N_trials_tr x N_neurons]
distTr = zeros(N_tr); % square matrix for mutual distance between training points
for t1 = 1:N_tr
    for t2 = 1:N_tr
        distTr(t1, t2) = sqrt(sum( (train_rates(t1, :)-train_rates(t2, :)).^2));
    end
end


%%  find predicted angle by selecting nearest neighbours --> Training
[sortedDistTr, indTr] = sort(distTr, 2);

nearest_tr = indTr(:, 2:n_neighbours+1);
nearest_dir_tr= zeros(N_tr, n_neighbours);
predicted_dir_tr= zeros(1, N_tr);
for row = 1:N_tr
    nearest_dir_tr(row, 1:n_neighbours) = labels_tr(nearest_tr(row, :));
    predicted_dir_tr(row) = mode(nearest_dir_tr(row, :));
end

diff = labels_tr - predicted_dir_tr;

% plot([1:N_tr]/N_tr*(N_angles), diff);
% xlabel("Labels (correct dir)");
% ylabel("DIfference predicted vs real");
% xlim([0,N_angles]);
% grid on

%% Try with testing data --> prediction

test_rates = zeros(N_angles*N_trials_test, N_neurons);
labels_test = zeros(1, N_angles*N_trials_test);
for k_it = 1:N_angles
    for n_it = 1:N_trials_tr
        test_rates( (k_it-1)*N_trials_tr + n_it, :) = sum(training_data(n_it, k_it).spikes(:, 1:t_pre_mvt), 2)';           
        labels_test( (k_it-1)*N_trials_tr + n_it) = k_it;
    end
end

[N_test, M_test] = size(test_rates);
distTest = zeros(N_test, N_tr);

% find distance to points from train data
for t1 = 1:N_test
    for t2 = 1:N_tr
        distTest(t1, t2) = sqrt(sum( (train_rates(t1, :)-train_rates(t2, :)).^2));
    end
end

[sortedDistTest, indTest] = sort(distTest, 2);

nearestTest = indTest(:, 2:n_neighbours+1);
nearest_dir_test= zeros(N_tr, n_neighbours);
predicted_dir_test= zeros(1, N_tr);
k_deduced = zeros(1, N_tr);

for row = 1:N_test
    nearest_dir_test(row, 1:n_neighbours) = labels_tr(nearestTest(row, :)); % the labels corresponding to the ones in the testing list
    predicted_dir_test(row) = mode(nearest_dir_test(row,1:n_neighbours));
    
    % try with average of vectors of nearest vects
    nearest_vects = zeros(2, n_neighbours);
    for n=1:n_neighbours
        nearest_vects(: , n) = unit_vect_list(:, nearest_dir_test(row, n));
    end
    predicted_vect = mean(nearest_vects, 2);
    predictedAng = atan(predicted_vect(2)/predicted_vect(1));
%     if predictedAng<0 % rectify angle
%         predictedAng = predictedAng +2*pi;
%     end
    k_deduced(row) = (180*predictedAng/pi +10)/40;
    
end


subplot(4,5, n_neighbours);
hold on;
plot([1:N_test]/N_test*(N_angles),predicted_dir_test, 'DisplayName','knn deduction');
plot([1:N_test]/N_test*(N_angles), k_deduced, 'DisplayName','using dir vectors');
lgd = legend; lgd.Location = 'northwest';

meanSqError(n_neighbours) = mean((predicted_dir_test - labels_test).^2)


