noTrials = 50; % number of trials to include - section for training data 
binWidth = 5; 
[all_nrns, pos_time_axis] = trial_averaged_neurons(noTrials, binWidth); % create averages across all trials of each neuron in each direction 
    % time axis points taken as centre of bin 

%% X-Direction analysis 
axis = 1; % X direction movement 
axis = 2; % Y direction movement

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
 
figure % plot x-movement for each direction, highlighting 'movement' section
for direction = [1:1:8]
    subplot(4, 2, direction)
    plot(pos_time_axis{direction}, all_psns{direction})
    hold on
    plot(pos_time_axis{direction}(ceil(299/binWidth):end-floor(100/binWidth)), all_psns{direction}(ceil(299/binWidth):end-floor(100/binWidth)))
    title("Direction " + direction)
    xlabel('Time (ms)')
    ylabel('X-Position (cm')
end

%% X-direction velocity tuning curves - set up data to create fitted tuning curve

% option to use all data, data from first 300ms, from movement period, or
% last 100 ms

tuning_curve_data = cell(98,8);
for neuron = [1:1:98]
    for direction = [1:1:8]
        tuning_curve_temp = [];

        % create cell array for tuning curve data - allows for differing data lengths 
        % first row  in each cell will contain x-velocity data, sorted in increasing
        % magnitude of position 
        % second row will contain corresponding spike data 
        
        tuning_curve_temp(1,:) = all_vels{direction}; % x-velocity data
        tuning_curve_temp(2,:) = all_nrns{neuron, direction}; % corresponding spike data
        [~, order] = sort(tuning_curve_temp(1, :)); % sort 
        tuning_curve_temp = tuning_curve_temp(:, order);
       
        tuning_curve_data{neuron, direction} = tuning_curve_temp(1:2,:); 

%         plot(tuning_curve_data((2*direction)-1,:), tuning_curve_data(2*direction,:),'.')
%         hold on 
    end
end

pre_300ms_tuning_curve_data = cell(98,8);
for direction = [1:1:8]
    for neuron = [1:1:98]
        tuning_curve_temp = [];

        % create cell array for tuning curve data - allows for differing data lengths 
        % first row  in each cell will contain x-position data, sorted in increasing
        % magnitude of position 
        % second row will contain corresponding spike data 
        
        tuning_curve_temp(1,:) = all_vels{direction}(1:floor(299/binWidth)); % x-position data
        tuning_curve_temp(2,:) = all_nrns{neuron, direction}(1:floor(299/binWidth)); % corresponding spike data
        [~, order] = sort(tuning_curve_temp(1, :)); % sort 
        tuning_curve_temp = tuning_curve_temp(:, order);
       
        pre_300ms_tuning_curve_data{neuron, direction} = tuning_curve_temp(1:2,:); 

%         plot(tuning_curve_data((2*direction)-1,:), tuning_curve_data(2*direction,:),'.')
%         hold on 
    end
end

movement_tuning_curve_data = cell(98,8);
for direction = [1:1:8]
    for neuron = [1:1:98]
        tuning_curve_temp = [];

        % create cell array for tuning curve data - allows for differing data lengths 
        % first row  in each cell will contain x-position data, sorted in increasing
        % magnitude of position 
        % second row will contain corresponding spike data 
        
        tuning_curve_temp(1,:) = all_vels{direction}(ceil(299/binWidth):end-floor(100/binWidth)); % x-position data
        tuning_curve_temp(2,:) = all_nrns{neuron, direction}(ceil(299/binWidth):end-floor(100/binWidth)); % corresponding spike data
        [~, order] = sort(tuning_curve_temp(1, :)); % sort 
        tuning_curve_temp = tuning_curve_temp(:, order);
       
        movement_tuning_curve_data{neuron, direction} = tuning_curve_temp(1:2,:); 

%         plot(tuning_curve_data((2*direction)-1,:), tuning_curve_data(2*direction,:),'.')
%         hold on 
    end
end

post_movement_tuning_curve_data = cell(98,8);
for direction = [1:1:8]
    for neuron = [1:1:98]
        tuning_curve_temp = [];

        % create cell array for tuning curve data - allows for differing data lengths 
        % first row  in each cell will contain x-position data, sorted in increasing
        % magnitude of position 
        % second row will contain corresponding spike data 
        
        tuning_curve_temp(1,:) = all_vels{direction}(end-ceil(100/binWidth):end); % x-position data
        tuning_curve_temp(2,:) = all_nrns{neuron, direction}(end-ceil(100/binWidth):end); % corresponding spike data
        [~, order] = sort(tuning_curve_temp(1, :)); % sort 
        tuning_curve_temp = tuning_curve_temp(:, order);
       
        post_movement_tuning_curve_data{neuron, direction} = tuning_curve_temp(1:2,:); 

%         plot(tuning_curve_data((2*direction)-1,:), tuning_curve_data(2*direction,:),'.')
%         hold on 
    end
end
%% Visualise Tuning Curve 
% uncomment relevant desired time period
% tuning_curve_data = tuning_curve_data;
% tuning_curve_data = pre_300ms_tuning_curve_data;
tuning_curve_data = movement_tuning_curve_data;
% tuning_curve_data = post_movement_tuning_curve_data;


% Fit used is 'gauss2' - the mean is used as the preferred direction. Plots
% compare first mean b1 to second mean b2 - first mean looks more
% reasonable to use - this may need more investigation/tuning - at the very
% least it will require re-writing so as not to use a package 

% array is created of preferred x-velocities of each neuron when movement is in each of the
% 8 directions

preferred_vel_gauss2 = [];
for direction = [1:1:8]
%     figure()
%     ax = [];
    for neuron = [1:1:98]
        
        f = fit(tuning_curve_data{neuron, direction}(1,:).',tuning_curve_data{neuron, direction}(2, :).','gauss2');
        preferred_vel_gauss2(neuron, direction) = f.b1;
        
%         % uncomment to plot/visualise
%         ax(neuron) = subplot(7,14, neuron);        
%         plot(tuning_curve_data{neuron, direction}(1,:), tuning_curve_data{neuron, direction}(2, :), '.', 'DisplayName', 'Spike Data')
%         hold on
%         xlabel("x velocity")
%         ylabel("Spike Frequency (Hz)")
%         plot([f.b1 f.b1], [1 150]);
%         plot([f.b2 f.b2], [1 150]);
%         title("Neuron " + neuron)
%         
    end
%    sgtitle("X-Velocity vs Spike Rate For Each Neuron: Direction = " + direction)

end

if axis == 1
    preferred_x_velocity = median(preferred_vel_gauss2,2);
else
    preferred_y_velocity = median(preferred_vel_gauss2, 2);
end

%% Y direction - either run above sections chnaging axis, or code repeated again below 

axis = 2; % Y direction movement

all_psns = trial_averaged_hand_pos(axis, noTrials, binWidth); 

all_vels = cell(1,8); % initialise cell array for velocity
vel_time_axis = cell(1, 8); % initialise cell array for velocity time points

% calculate hand velocity over time 
% time points for velocity taken as midpoint of time interval over which
% velocity calculated
for direction = [1:1:8]
    all_vels{direction} = [0, diff(all_psns{direction})./diff(pos_time_axis{direction})];
    vel_time_axis{direction} = [1, (pos_time_axis{direction}(1:end-1) + pos_time_axis{direction}(2:end))/2];
end
 
figure % plot y-movement for each direction, highlighting 'movement' section
for direction = [1:1:8]
    subplot(4, 2, direction)
    plot(pos_time_axis{direction}, all_psns{direction})
    hold on
    plot(pos_time_axis{direction}(ceil(299/binWidth):end-floor(100/binWidth)), all_psns{direction}(ceil(299/binWidth):end-floor(100/binWidth)))
    title("Direction " + direction)
    xlabel('Time (ms)')
    ylabel('X-Position (cm')
end

tuning_curve_data = cell(98,8);
for neuron = [1:1:98]
    for direction = [1:1:8]
        tuning_curve_temp = [];

        % create cell array for tuning curve data - allows for differing data lengths 
        % first row  in each cell will contain y-velocity data, sorted in increasing
        % magnitude of position 
        % second row will contain corresponding spike data 
        
        tuning_curve_temp(1,:) = all_vels{direction}; % y-velocity data
        tuning_curve_temp(2,:) = all_nrns{neuron, direction}; % corresponding spike data
        [~, order] = sort(tuning_curve_temp(1, :)); % sort 
        tuning_curve_temp = tuning_curve_temp(:, order);
       
        tuning_curve_data{neuron, direction} = tuning_curve_temp(1:2,:); 

%         plot(tuning_curve_data((2*direction)-1,:), tuning_curve_data(2*direction,:),'.')
%         hold on 
    end
end

pre_300ms_tuning_curve_data = cell(98,8);
for direction = [1:1:8]
    for neuron = [1:1:98]
        tuning_curve_temp = [];

        % create cell array for tuning curve data - allows for differing data lengths 
        % first row  in each cell will contain y-velocity data, sorted in increasing
        % magnitude of position 
        % second row will contain corresponding spike data 
        
        tuning_curve_temp(1,:) = all_vels{direction}(1:floor(299/binWidth)); % y-velocity data
        tuning_curve_temp(2,:) = all_nrns{neuron, direction}(1:floor(299/binWidth)); % corresponding spike data
        [~, order] = sort(tuning_curve_temp(1, :)); % sort 
        tuning_curve_temp = tuning_curve_temp(:, order);
       
        pre_300ms_tuning_curve_data{neuron, direction} = tuning_curve_temp(1:2,:); 

%         plot(tuning_curve_data((2*direction)-1,:), tuning_curve_data(2*direction,:),'.')
%         hold on 
    end
end

movement_tuning_curve_data = cell(98,8);
for direction = [1:1:8]
    for neuron = [1:1:98]
        tuning_curve_temp = [];

        % create cell array for tuning curve data - allows for differing data lengths 
        % first row  in each cell will contain x-position data, sorted in increasing
        % magnitude of position 
        % second row will contain corresponding spike data 
        
        tuning_curve_temp(1,:) = all_vels{direction}(ceil(299/binWidth):end-floor(100/binWidth)); % x-position data
        tuning_curve_temp(2,:) = all_nrns{neuron, direction}(ceil(299/binWidth):end-floor(100/binWidth)); % corresponding spike data
        [~, order] = sort(tuning_curve_temp(1, :)); % sort 
        tuning_curve_temp = tuning_curve_temp(:, order);
       
        movement_tuning_curve_data{neuron, direction} = tuning_curve_temp(1:2,:); 

%         plot(tuning_curve_data((2*direction)-1,:), tuning_curve_data(2*direction,:),'.')
%         hold on 
    end
end

post_movement_tuning_curve_data = cell(98,8);
for direction = [1:1:8]
    for neuron = [1:1:98]
        tuning_curve_temp = [];

        % create cell array for tuning curve data - allows for differing data lengths 
        % first row  in each cell will contain x-position data, sorted in increasing
        % magnitude of position 
        % second row will contain corresponding spike data 
        
        tuning_curve_temp(1,:) = all_vels{direction}(end-ceil(100/binWidth):end); % x-position data
        tuning_curve_temp(2,:) = all_nrns{neuron, direction}(end-ceil(100/binWidth):end); % corresponding spike data
        [~, order] = sort(tuning_curve_temp(1, :)); % sort 
        tuning_curve_temp = tuning_curve_temp(:, order);
       
        post_movement_tuning_curve_data{neuron, direction} = tuning_curve_temp(1:2,:); 

%         plot(tuning_curve_data((2*direction)-1,:), tuning_curve_data(2*direction,:),'.')
%         hold on 
    end
end
% % uncomment relevant desired time period
% tuning_curve_data = tuning_curve_data;
% tuning_curve_data = pre_300ms_tuning_curve_data;
tuning_curve_data = movement_tuning_curve_data;
% tuning_curve_data = post_movement_tuning_curve_data;


% Fit used is 'gauss2' - the mean is used as the preferred direction. Plots
% compare first mean b1 to second mean b2 - first mean looks more
% reasonable to use

% array is created of preferred x-velocities of each neuron when movement is in each of the
% 8 directions

preferred_vel_gauss2 = [];
for direction = [1:1:8]
%     figure()
%     ax = [];
    for neuron = [1:1:98]
        
        f = fit(tuning_curve_data{neuron, direction}(1,:).',tuning_curve_data{neuron, direction}(2, :).','gauss2');
        preferred_vel_gauss2(neuron, direction) = f.b1;
        
%         % uncomment to plot/visualise
%         ax(neuron) = subplot(7,14, neuron);        
%         plot(tuning_curve_data{neuron, direction}(1,:), tuning_curve_data{neuron, direction}(2, :), '.', 'DisplayName', 'Spike Data')
%         hold on
%         xlabel("x velocity")
%         ylabel("Spike Frequency (Hz)")
%         plot([f.b1 f.b1], [1 150]);
%         plot([f.b2 f.b2], [1 150]);
%         title("Neuron " + neuron)
%         
    end
%    sgtitle("X-Velocity vs Spike Rate For Each Neuron: Direction = " + direction)

end

if axis == 1
    preferred_x_velocity = median(preferred_vel_gauss2,2);
else
    preferred_y_velocity = median(preferred_vel_gauss2, 2);
end




