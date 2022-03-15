%% BMI Coursework
%% March 2022
%% Aline Buat

clear;
close all;


%% Init
load('monkeydata_training.mat');

%% make a subset of original data for training
trial_tr = trial([1:20,41:100], :); % subset of original data set
trial_test = trial(21:40, :);  % subset of original data set for testing

% dimentions of data
N_trials = size(trial_tr, 1);
N_reaching_angles = size(trial_tr, 2);
N_neuralunits = size(trial_tr(1,1).spikes, 1);
N_trial_tr = size(trial_tr, 1);

% make a list of unit direction vectors for each angle (cartesian, x, y)
k_list = 1:N_reaching_angles;
theta = (40*k_list-10)/180*pi;
unit_vect_list = [cos(theta); sin(theta)];


%% tuning curve
% Angle from 1st 300 ms


n_unit = 70;
N_bins = 25;


k_list = 1:N_reaching_angles;
color_list = {'#0072BD', '[0.8500 0.3250 0.0980]','[0.4940 0.1840 0.5560]',...
'[0.9290 0.6940 0.1250]','r','g','b','m'};

neuron_tuning = zeros(1, N_reaching_angles);
figure; hold on;
xlabel('t [ms]'); ylabel('Firing rate [Hz]');



for k = k_list
    [neuron_tuning_dir_k_mean, neuron_tuning_dir_k_std] = neuron_tuning_dir_k(trial, n_unit, k, N_bins);
    neuron_tuning(k) = neuron_tuning_dir_k_mean;
    neuron_tuning_std(k) = neuron_tuning_dir_k_std;
    fprintf("Dir %g: av spike rate=%g \n", k, neuron_tuning_dir_k_mean);

end
legend;

theta = (30+40*k_list)/180*pi;
figure;
polarplot([theta,theta(1)], [neuron_tuning,neuron_tuning(1)],'-.o', 'MarkerSize', 8);
title("Tuning Curve of Neuron unit ="+n_unit);


figure;grid on; hold on;
errorbar(k_list, neuron_tuning, neuron_tuning_std);
xlabel('k');
ylabel('Average firing rate');
title("Neuron unit ="+n_unit);
ylim([0,inf]);

% f_tuning = fit(k_list', neuron_tuning', 'gauss1');
% plot(f_tuning);
f_smooth = smoothdata(neuron_tuning, 'gaussian', 4);
plot(f_smooth);


pdf_tuning = neuron_tuning/sum(neuron_tuning, 'all');
figure;
plot(pdf_tuning);
ylim([0,inf]);


%% Encode Population tuning

k_list = 1:N_reaching_angles;
n_list = 1:N_trial_tr;
neuron_list = 1:N_neuralunits;

theta = (30+40*k_list)/180*pi;


neuron_tuning = zeros(N_trial_tr, N_reaching_angles);

figure; 
for neuron = neuron_list
    for k = k_list
        [neuron_tuning_dir_k_mean, neuron_tuning_dir_k_std] = neuron_tuning_dir_k(trial_tr, neuron, k, N_bins);
        neuron_tuning(neuron, k) = neuron_tuning_dir_k_mean;
    end
    neuron_tuning(neuron,:)=smoothdata(neuron_tuning(neuron,:), 'gaussian', 3.5);
    polarplot([theta, theta(1)], [neuron_tuning(neuron,:), neuron_tuning(neuron,1)]); hold on;
end
title('Tuning Curves for all neurons');

figure;
plot(theta, neuron_tuning(1:5:end,:)); hold on;
set(gca,'XTick',0:pi/2:2*pi) 
set(gca,'XTickLabel',{'0','\pi/2','\pi','3\pi/2','2\pi'})
xlabel('\theta');
ylabel('Neuron Response');
title('Tuning Curves for all neurons');
xlim([theta(1)-0.1,theta(end)+0.1]);

%% test on data to predict direction of movement

% select trial to test
N_trial_test = size(trial_test, 1);

n = 11;
k= floor(rand*8)+1;
% k = 7;
% t = floor(rand*size(spikes, 2)) + 1;
t= 300;
neuron_list = 1:N_neuralunits;

spikes = trial_test(n, k).spikes;


spikes = spikes(:, 1:t); % try on limited time data 

% get average rate for each neuron
neuron_resp = sum(spikes, 2)/size(spikes,2)*1000;

% initialize population difference

[Max_tune, I] = max(neuron_tuning, [], 2);
Min_tune = min(neuron_tuning, [],2);
fa = (neuron_resp-Min_tune)./ (Max_tune-Min_tune);   % size 98x1

% vector 98x2 contraining directions of tuning
tune_vect_list = unit_vect_list(:,I)'; 

% orientation_vect = ones(1, size(fa,1))*(fa.*tune_vect_list);
orientation_vect = sum(fa.*tune_vect_list,1);
orientation_vect = orientation_vect ./ sqrt(orientation_vect(1)^2+orientation_vect(2)^2)

dir = atan(orientation_vect(2)/orientation_vect(1))



k_deduced = (180*dir/pi + 10)/40
k

fprintf("Error between real k and k deduced: %g\n", k-k_deduced);


%% several trials
k_list = 1:N_reaching_angles;
n_list = 1:N_trial_test;
t_list = [500];

figure; hold on; grid on

k_deduced = zeros(size(n_list,1), size(k_list,1));
dir = zeros(size(n_list,1), size(k_list,1));

for k =k_list
    for n = n_list
        spikes = trial_test(n,k).spikes;
        dir(n,k) = position_estimation(spikes, neuron_tuning, unit_vect_list);
        if dir(n,k)<0
            dir(n,k) = dir(n,k)+2*pi;
        end
        k_deduced(n,k) = (180*dir(n,k)/pi + 10)/40;
    end 
    plot(n_list, abs(k_deduced(:,k)), 'DisplayName', "k="+k); hold on;
end
xlabel('n');
ylabel('Estimation of k');


legend;



plot(n_list, abs(k_deduced))