clear
load('monkeydata_training.mat')

spikes_all = trial.spikes;
handPos_all = trial.handPos;

% 'The data contains spike trains recorded from 98 neural units while the
% monkey reached 182 times along each of 8 different reaching angles, 
% as well as the monkey’s arm trajectory on each trial'

% 'The set has been divided into a training dataset of 100 trials per 
% reaching angle'

I = size(spikes_all,1);
N = size(trial,1);
T = size(spikes_all,2);
K = size(trial,2);

%% Tuning curves for several neurons
close all

start_t = 300;
end_t = -100;

legend_array = [];
tunings_array = [];
figure

colours = ['b','r','g','y','m','c','k',[0 0.4470 0.7410],[0.8500 0.3250 0.0980],[0.9290 0.6940 0.1250]];
l = 1;
for i=1:I
    tunings = p2t(trial,i);
    %tunings_all(i,:) = tunings;

    % threshold - include to plot only neurons that meet threshold
    tunings_range = max(tunings) - min(tunings);
%     if (tunings_range > 0.005) && (min(tunings) > 0.02)
%         legend_array = [legend_array; i];
%         tunings_array = [tunings_array; tunings];
%         plot(1:K, tunings,colours(l),'DisplayName', num2str(i));
%         l = l+1;
%         hold on
%     end

    % original - include to plot all 98 neurons
    legend_array = [legend_array; i];
    tunings_array = [tunings_array; tunings];
    plot(1:K, tunings,'DisplayName', num2str(i));
    hold on
end
legendStrings = "i = " + string(legend_array);
legend(legendStrings)
title('All tuning curves')

%% K-means clustering
% Hierarchical won't work as there are 98 neurons to look through - (98!)
% iterations would be required
close all
% intiialise centroids - dummy neuron analogy
clusters = 10;
min_tuning = min(tunings_array,[],1);
max_tuning = max(tunings_array,[],1);
centroids = zeros(clusters,K);
for c=1:clusters
    for k=1:K
        centroids(c,k) = (max_tuning(k)-min_tuning(k)).*rand(1)+min_tuning(k);
    end
end
centroids;
% original centroids (randomised coordinates)
figure
for c=1:clusters
    plot(1:K,centroids(c,:),colours(c),'LineWidth',2)
    hold on
end

% assign centroids to each neuron
assigned_cluster = zeros(I,1);
figure
for iter=1:8
    centroids;
    for i=1:I
        centroid_dist = zeros(clusters,1); % temporarily store squared difference between each neuron and centroid
        for c=1:clusters
            for k=1:8
                % summed across directions for each centroid
                centroid_dist(c) = centroid_dist(c) + (tunings_array(i,k)-centroids(c,k))^2;
            end
        end
        cluster_no = find(centroid_dist==min(centroid_dist));
        i;
        assigned_cluster(i) = cluster_no;
    end
    assigned_cluster;
    
    subplot(5,2,iter)
    for i=1:98
        %scatter(1:K, tunings_array(i,:), colours(assigned_cluster(i)),'filled')
        plot(1:K, tunings_array(i,:), colours(assigned_cluster(i)))
        hold on
    end
    for c=1:clusters
        %scatter(1:K, centroids(c,:), 200, colours(c), 'filled')
        plot(1:K, centroids(c,:), colours(c), 'LineWidth', 2)
        hold on
    end
    
    %Calculate group means and update to centroids
    for c=1:length(clusters)
        members = find(assigned_cluster==c);
        centroids(c,:) = mean(tunings_array(members,:));
    end
end

features = zeros(clusters,1);
for c=1:clusters
    members = find(assigned_cluster==c);
    ranges = zeros(size(members));
    for m=1:length(members)
        m;
        ranges(m) = max(tunings_array(members(m),:))-min(tunings_array(members(m),:));
    end
    ranges;
    chosen_one = find(ranges==max(ranges));
    if (isempty(chosen_one))
        features(c) = 0;
    else
        features(c) = members(chosen_one);
    end
end

% Delete zero elements in features
features = features(find(features~=0));



%%
j=4

load('monkeydata_training.mat')

noTrials = 80; % number of trials to include - section for training data 
binWidth = 5; 
[all_nrns_a, pos_time_axis] = trial_averaged_neurons(noTrials, binWidth); % create averages across all trials of each neuron in each direction 
[test_data,~] = trial_averaged_neurons(noTrials, binWidth);
all_nrns = cell(length(features),8);
for f = 1:1:length(features)
    for direction=[1:1:8]
        all_nrns{f, direction} = all_nrns_a{features(f),direction};
    end
end


% axis = 1; % X direction movement 
axis = 2; % Y direction movement

all_psns = trial_averaged_hand_pos(axis, noTrials, binWidth); 
    % average hand position taken over all trials

all_vels = cell(1,8); % initialise cell array for velocity
vel_time_axis = cell(1, 8); % initialise cell array for velocity time points
% velocity calculated (Meghan)

for direction = [1:1:8]
    all_vels{direction} = [0, diff(all_psns{direction})./diff(pos_time_axis{direction})];
    vel_time_axis{direction} = [1, (pos_time_axis{direction}(1:end-1) + pos_time_axis{direction}(2:end))/2];
end


% for j=1:1:8

a=[]
    for k = 1:length(features)
        for t = 1:1:length(all_vels{j})
%         p = p + x_vel_weights(k)*trial(100,j).spikes(k,i); % summation part of LR formula
%             u(:, k) = trial(100,j).spikes(k,:); % u vector from LR formula 
            a(k, t) = all_nrns{k,j}(t);
        end
%     end

     
    %     y_lr = lsqminnorm(spike_train_binned(:,:,j), y_vel_points(:,j));
%         model1(j).xlr=x_lr;
    %     model2(j).ylr=y_lr;
    end
% end
a = transpose(a);
all_vels{j} = transpose(all_vels{j})
size(a)
size(all_vels{j})
   x_lr =lsqminnorm(a, all_vels{j}); %getting coefficients

%%
u = [];
p = 0;

load('monkeydata_training.mat')
pred_x_vel = [];
testTrial = 100
for i = [1:1:length(trial(100,j).spikes(1,:))]  % loop through all time points
    for k = 1:length(features)
%         p = p + x_vel_weights(k)*trial(100,j).spikes(k,i); % summation part of LR formula
        u(:, k) = trial(100,j).spikes(k,:); % u vector from LR formula 
    end
    pred_x_vel(i) = transpose(x_lr)* transpose(u(i,:));
end
act_x_vel = [0, diff(trial(testTrial,j).handPos(axis,:))./diff([1:1:length(trial(testTrial,j).handPos(axis,:))])];

figure
plot(pred_x_vel)
hold on 
plot(act_x_vel)
xlabel("Time (ms)")
ylabel("X Velocity")
title("Direction k = " + j)
legend(['Predicted: Linear Regression'], [("Actual Position: Trial " + testTrial)])

function tunings = p2t(trial,i)
    K=8;
    N=100;
    T_max=975;
    spikes_padded = zeros(K, N, T_max);

    for k=1:K
        for n=1:N
            spikes_cur = trial(n,k).spikes(i,:);
            T_cur = size(spikes_cur,2);
            if (T_cur<T_max)
                spikes_padded(k,n,:) = cat(2, spikes_cur, zeros([1 T_max-T_cur]));
            end
        end
    end
    
    total_spikes = sum(spikes_padded,3);
    rates = total_spikes/T_max;
    rates_smooth = smoothdata(rates,'gaussian', 3);
    tunings = zeros(1,K);
    for k=1:8
        tunings(k) = mean(rates_smooth(k,:));
    end
end
