function [modelParameters] = positionEstimatorTraining(training_data)
  % Arguments:
  
  % - training_data:
  %     training_data(n,k)              (n = trial id,  k = reaching angle)
  %     training_data(n,k).trialId      unique number of the trial
  %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
  %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
  
  % ... train your model


    % Initialisation

    load('monkeydata_training.mat');
    spikes_all = trial.spikes;
    handPos_all = trial.handPos;
    
    I = size(spikes_all,1); % neurons
    N = size(trial,1); % trials
    T = size(spikes_all,2); % time steps
    K = size(trial,2); % directions

    % Find tuning curves    

    start_t = 300;
    end_t = -100;
    
    legend_array = [];
    tunings_array = [];

    for i=1:I
        tunings = p2t(trial,i);
    
        % threshold - include if statement to plot only neurons that meet threshold
        range = max(tunings) - min(tunings);
    %     if (range > 0.005) && (min(tunings) > 0.02)
        legend_array = [legend_array; i];
        tunings_array = [tunings_array; tunings];
    %     end
    end

    % K-means clustering

    % intiialise centroids - dummy neuron analogy
    clusters = 5;
    min_tuning = min(tunings_array,[],1);
    max_tuning = max(tunings_array,[],1);
    centroids = zeros(clusters,K);
    for c=1:clusters
        for k=1:K
            centroids(c,k) = (max_tuning(k)-min_tuning(k)).*rand(1)+min_tuning(k);
        end
    end

    % assign centroids to each neuron
    assigned_cluster = zeros(I,1);
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
        
        %Calculate group means and update to centroids
        for c=1:length(clusters)
            members = find(assigned_cluster==c);
            centroids(c,:) = mean(tunings_array(members,:));
        end
    end
    
    % find prominent neurons
    features = zeros(clusters,1);
    for c=1:clusters
        members = find(assigned_cluster==c);
        ranges = zeros(size(members));
        for m=1:length(members)
            m;
            ranges(m) = max(tunings_array(members(m),:))-min(tunings_array(members(m),:));
        end
        chosen_one = find(ranges==max(ranges));
        if (isempty(chosen_one))
            features(c) = 0;
        else
            features(c) = members(chosen_one);
        end
    end

    % Regression

  % Return Value:
  
  % - modelParameters:
  %     single structure containing all the learned parameters of your
  %     model and which can be used by the "positionEstimator" function.
  
end

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