noTrials = 50; % number of trials to include - section for training data 
binWidth = 5; 
% [all_nrns, time_axis] = trial_averaged_neurons(noTrials, binWidth); % create averages across all trials of each neuron in each direction 
all_nrns = trial_averaged_neurons(noTrials, binWidth); % create averages across all trials of each neuron in each direction 

axis = 1; % X direction movement 
all_psns = trial_averaged_hand_pos(axis, noTrials, binWidth); 

% Neurons could be responsible for direction or acceleration?

%% PSTH 
figure
direction = 8;
neuron = 5;
time_axis = ([1:1:size(all_nrns{neuron, direction},2)]*binWidth)/1000;
binCenters = time_axis - 1/1000 + (binWidth/2000);
bar(binCenters, all_nrns{neuron, direction})
title("PSTH for Neuron " + neuron + " Direction k = " + direction)
ylabel("Spike Frequency (Hz)")
xlabel("Time (ms)")
%% X-direction tuning curves - set up data to create fitted tuning curve 

tuning_curve_data = cell(98,8);
for direction = [1:1:8]
    for neuron = [1:1:98]
        tuning_curve_temp = [];

        % create cell array for tuning curve data - allows for differing data lengths 
        % first row  in each cell will contain x-position data, sorted in increasing
        % magnitude of position 
        % second row will contain corresponding spike data 
        
        tuning_curve_temp(1,:) = all_psns{direction}; % x-position data
        tuning_curve_temp(2,:) = all_nrns{neuron, direction}; % corresponding spike data
        [~, order] = sort(tuning_curve_temp(1, :)); % sort 
        tuning_curve_temp = tuning_curve_temp(:, order);
       
        tuning_curve_data{neuron, direction} = tuning_curve_temp(1:2,:); 

%         plot(tuning_curve_data((2*direction)-1,:), tuning_curve_data(2*direction,:),'.')
%         hold on 
    end
end

%% Code here used to visualise fit of polynomials of different order. n = 12 seems a reasonable compromise for use for tuning curves
direction = 1;
i = 1;
figure
for neuron = [1:1:98]
    if ismember(neuron, [15, 29, 43, 57, 71, 85])
        figure
        i = 1;
    end
    subplot(3, 5, i)
    plot(tuning_curve_data{neuron, direction}(1,:), tuning_curve_data{neuron, direction}(2, :), '.', 'DisplayName', 'Spike Data')
    hold on
    for n = [1:5:30] % different orders for polynomial fit
        yFit = zeros(1,size(tuning_curve_data{neuron, direction}, 2));
        counter = 1;
        index = n;
        p = polyfit(tuning_curve_data{neuron, direction}(1,:),tuning_curve_data{neuron, direction}(2,:),n);
        while index >= 0
            yFit = yFit + tuning_curve_data{neuron, direction}(1,:).^index*p(counter);
            index = index - 1;
            counter = counter + 1;
        end
        plot(tuning_curve_data{neuron, direction}(1,:), yFit(:), 'DisplayName', "Polynomial Fit Order " + n)
        hold on 
    end
    xlabel("x position")
    ylabel("Spike Frequency (Hz)")
    title("Neuron " + neuron)
    ylim([0 70])
    legend
    i = i + 1;
end


%% Create fit curves for tuning curves with n = 12
fitted_tuning_curves = cell(98, 8);
n = 12; % polynomial fit order
for direction = [1:1:8]
    for neuron = [1:1:98]
        yFit = zeros(1,size(tuning_curve_data{neuron, direction}, 2));
        counter = 1;
        index = n;
        p = polyfit(tuning_curve_data{neuron, direction}(1,:),tuning_curve_data{neuron, direction}(2,:),n);
        while index >= 0
            yFit = yFit + tuning_curve_data{neuron, direction}(1,:).^index*p(counter);
            index = index - 1;
            counter = counter + 1;
        end
        fitted_tuning_curves{neuron, direction} = yFit;
    end
end
%% 
%% ARCHIVE 
for direction = [1:1:8]
    figure 
    title("Relationship Between Neuron Spike Rate and X-Direction Position: k = " + direction)
    i = 1;
    for neuron = [1:1:98]
        subplot(14, 7, i)
        yyaxis left 
        plot(time_axis, all_nrns(((direction - 1)*98)+neuron,:))
        ylabel("Expected Number of Spikes in " + binWidth + "ms")
        yyaxis right 
        plot(time_axis, all_psns(direction,:))
        ylabel("Hand Position in X Direction")
        xlabel('Time (ms)') 
        title("Neuron: " + neuron)
        i = i + 1;
    end
end
%     