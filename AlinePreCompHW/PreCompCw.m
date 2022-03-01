%% Pre-Task for BMI competition
%% Aline Buat
% 26/02/2022

clear
close all;


% Description of the data:
%   spike trains recorded from 98 neural units
%   monkey reached 182 times along each of 8 different reacing angles
%   monkey's arm trajectory on each trial

% both neural data and arm trajectory are taken from 300ms before movement
% onset until 100ms after mvt ends


% .mat file:
% single variable called trial size:
% 100(trials) x 8 (reaching angles) 
% fields in each: 
% time steps (num of cols) 1ms
%   trialID     % just a number
%   spikes      % neural data -- taken from 98 neural units over varying
%                 time periods
%   handPos     % hand position traces (x, y, z) for each t

%% Start

load('monkeydata_training.mat');


% trial(n,k) - n = nth trial , k = k reaching angle
% trial(n,k).spikes(i,t) - ith -neural unit (out of 98) 
% t-time in ms (variable size)

% n=1:100
% k=1:8
% i=1:98

N_trials = size(trial,1); % = 100
N_reaching_angles = size(trial, 2); % = 8
N_neuralunits = 98; % given


%% Plot Hand Position vs reaching angle
figure;
n_unit = 1;  % trial num
k = 1;  % reaching angles (from 1:8), 30pi/180,

for k = 1:8
    ang = (30+40*k)/180*pi;
    subplot(2,4,k);
    %plot(trial(n_unit,k).handPos');
    scatter3(trial(n_unit,k).handPos(1,:), trial(n_unit,k).handPos(2,:), trial(n_unit,k).handPos(3,:),'filled', 'ColorVariable', 'x');
    title_text = ["Trial "+ n_unit, "Reaching angle="+(30+40*k)+"\pi /180"];
    title(title_text);
    xlabel('x');
    ylabel('y');
    zlabel('z');
%     xlabel('t [ms]');
%     ylabel('Arm trajectory');
   % legend({'x', 'y', 'z'}, 'Location', 'northwest');

end

%% Raster plot for a single trial: display a population raster plot
% time (bins) on x-axis, neural units on y-axis

figure;
n=20;
k=1;


for k = 1:N_reaching_angles
    subplot(N_reaching_angles,1,k); hold on;
    spikes = trial(n, k).spikes;
    t_max = size(spikes, 2);
    y=[];
    for t = 1:t_max
        pos = find(spikes(:,t)==1);
        neg = find(spikes(:,t)==0);
        plot(ones(size(pos)).*t, pos, 'b.');
    end

    ylabel(["k="+k]);

end

xlabel("Time [ms]");


k = round(N_reaching_angles/2);
subplot(N_reaching_angles, 1, k);
ylabel(["Neuron Number", "k="+k]);
subplot(N_reaching_angles,1,1);
title(["Raster plot of neuron activity for a single trial"], ...
        "n=" + n);



%% raster plot for one neural unit over many trials

n_list = 1:N_trials;
k = 1;
n_unit = 1; % neural unit selected

figure;
hold on;

for n = n_list
    spikes_neuron = trial(n, k).spikes(n_unit,:);
    pos = find(spikes_neuron == 1);
    plot(pos, ones(size(pos))*n, '.');
end
ylabel("Trial");
xlabel("Time [ms]");

title("Raster plot of one neural unit over many trials, k="+k+", Neural unit: "+n_unit);


%% Peri-Stimulus histograms (PSTHs)

n_list = 1:N_trials;    % 
k = 1;  % reaching angle selected
n_unit = 25; % neural unit selected

N_bins = 25;

spikes_neuron = concat_spikes_neuron_dir_k(trial, n_unit, k);

spikes_neuron_bins = sum(spikes_neuron, 1);
spikes_neuron_mean = mean(spikes_neuron, 1);
spikes_neuron_var = var(spikes_neuron, 1);

T = size(spikes_neuron, 2);

Delt = T / N_bins;
spikes_hist = zeros(1, N_bins);

for i = 1: N_bins
    sub_zone = spikes_neuron_mean(round(Delt*(i-1)+1) : round(Delt*i) );
    spikes_hist(i) = sum(sub_zone)/Delt*1000;
end


figure;
hold on;

% bar(Delt*[1:N_bins], spikes_hist);
% xlabel("Time [ms]");
% ylabel("Extimated Spike per ms");
% title("Neural unit: "+n_unit+", Reaching angle="+k );

figure;
edges = 1:Delt:Delt*(N_bins+1);
h1 = histogram('BinEdges',edges, 'BinCounts', spikes_hist);
title("Neural unit: "+n_unit+", Reaching angle="+k );
xlabel("Time [ms]");
ylabel("Extimated Spikes per ms");




x_values = 1:length(spikes_neuron_bins);
x_values2 = edges(1:end-1);
f = fit(x_values2', spikes_hist', 'gauss1');
hold on; plot(f)
xlabel("Time [ms]");
ylabel("Extimated Spike rate [Hz]");


%% Part 4: Hand position for diff trials

figure;
n_list = 1:N_trials;
k_list = 1:N_reaching_angles;

color_list = {'#0072BD', '[0.8500 0.3250 0.0980]','[0.4940 0.1840 0.5560]',...
'[0.9290 0.6940 0.1250]','r','g','b','m'};
label_list = strings(length(n_list)*length(k_list));

for k = k_list
    k_color = color_list{k};
    for n = n_list
        hold on;
        plot3(trial(n, k).handPos(1,:), trial(n, k).handPos(2,:), trial(n, k).handPos(3,:), 'Color',k_color);
    end
%     label_list(n*k) = "k="+k;
    label_list(n*k) = "\theta=^{"+num2str(30+40*k)+"\pi}/_{180}";
end
xlabel('x');
ylabel('y');
% zlabel('z');

legend(label_list,'Location','northwest');
title('Arm trajectories for different reaching angles, all trials')
grid on


%% Part 5

k_list = 1:N_reaching_angles;
n_list = 1: N_trials;

N_bins = 25;


n_unit = 30;
k = 1;
spikes_neuron = concat_spikes_neuron_dir_k(trial, n_unit, k);

% remove first 300 s without movement (2nd dimention)
spikes_neuron = spikes_neuron(:, 300:end);


T = size(spikes_neuron, 2);
Delt = T / N_bins;

spikes_neuron_mean = mean(spikes_neuron, 1);
spikes_neuron_var = var(spikes_neuron, 1);

% make bins over time:
spikes_hist = zeros(1, N_bins);
for i = 1: N_bins
    sub_zone = spikes_neuron_mean(floor(Delt*(i-1)+1) : floor(Delt*i) );
    spikes_hist(i) = sum(sub_zone)/Delt*1000; % gives rate in Hz vs time t
end

% plotting
figure; hold on;
edges = 1:Delt:Delt*(N_bins+1); % edges of the bins (in ms)
h1 = histogram('BinEdges',edges, 'BinCounts', spikes_hist);
title("Neural unit: "+n_unit+", Reaching angle="+k );
xlabel("Time [ms]");
ylabel("Extimated Spikes per ms");

x_values = edges(1:end-1);
f = fit(x_values', spikes_hist', 'gauss1');
hold on;
plot(f);

[f_smooth, window] = smoothdata(spikes_hist, 'gaussian', N_bins/2);
plot(x_values, f_smooth, 'DisplayName', 'smooth')

spikes_std_t = f.c1; % c1 is the standard deviation of the gauss fit

spikes_mean_t = f.b1; % b1 is the mean of the fit (in time)

neuron_mean_tune = mean(f_smooth);


%% part 5 with funtion


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

% Fit tuning curve

f_tuning = fit(theta', neuron_tuning', 'gauss1');
plot(f_tuning);




%% Multiplot of arm movement in 3D



figure;
n_list = 1:24:N_trials;
k_list = 1:N_reaching_angles;

color_list = {'#0072BD', '[0.8500 0.3250 0.0980]','[0.4940 0.1840 0.5560]',...
'[0.9290 0.6940 0.1250]','r','g','b','m'};
label_list = strings(length(n_list)*length(k_list));

full_handPos_concat=[];

% concatenate all hand positions into one massive array

for k = k_list
    k_color = color_list{k};
    % concatenate hand positions for 1 direction in an array
    handPos_concat = concat_hand_mvt_dir_k(trial, k, n_list);
    
    if isempty(full_handPos_concat)
        full_handPos_concat = handPos_concat; % copy into first position
    else
        % check if new size matches old
        sf = size(full_handPos_concat); % size of old
        sn = size(handPos_concat); % size of new
        if sn(2) ~= sf(2) 
            if sn(2) < sf(2)  % new smaller
                %pad = ones(sn(1), sf(2) - sn(2), sn(3)).*full_handPos_concat(:,end,:);
                handPos_concat(sn(1),sf(2),sn(3))=0; % expand the matrix with weird trick
            elseif sn(2) > sf(2) % new larger
                full_handPos_concat(sf(1),sn(2),sf(3)) = 0;
            end
        end
        full_handPos_concat = cat(3, full_handPos_concat, handPos_concat);
        fprintf("Concat successful k=%g\n",k);
    end
    n = n_list(end);       
    label_list(n*k) = "\theta=^{"+num2str(30+40*k)+"\pi}/_{180}";  
end

% handPos_concat = concat_hand_mvt_dir_k(trial, k, n_list);
X = full_handPos_concat(1,1:end);
Y = full_handPos_concat(2,1:end);
Z = full_handPos_concat(3,1:end);

plot3(X,Y,Z);
xlabel('x');
ylabel('y');
zlabel('z');

legend(label_list,'Location','northwest');
title('Arm trajectories for different reaching angles, all trials')
grid on



