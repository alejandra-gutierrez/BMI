%% Initialisation

close all
load('monkeydata_training.mat');
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
    range = max(tunings) - min(tunings);
%     if (range > 0.005) && (min(tunings) > 0.02)
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
for iter=1:6
    centroids
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
    
    subplot(3,2,iter)
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

%% PCA

% covariance C, eigenvectors V, eigenvalues D
tunings_array = normalize(tunings_array);
C = cov(tunings_array);
[V, D] = eig(C);
last = size(V,2);
% feature vector contains eigenvectors corresponding to highest eigenvalues
% (highest variance)
F = [V(:,last) V(:,last-1)];

% scree plot: magnitude of eigenvalues plotted against principle components
figure
plot(8:-1:1, max(D)')
xlabel('Principle Component')
ylabel('Eigenvalue')
title('Scree plot')

% 2D PCA plot - projected
new_tunings_array = transpose(F) * transpose(tunings_array);
figure
for i=1:I
    scatter(new_tunings_array(1,i), new_tunings_array(2,i), 'filled')
    hold on
end
legend(legendStrings)

new_C = cov(normalize(transpose(new_tunings_array)));
[new_V,new_D] = eig(new_C);
% new_V = -new_V;
% new_D = -new_D;
% new_C = -new_C;
for j=1:2
    plot([0 new_V(1,j)], [0 new_V(2,j)], 'LineWidth', 2)
    hold on
end
xlabel('PC1')
ylabel('PC2')
title('Projected Principle Components')

% 1D PCA plot
g_all = [];
figure
for l=1:2
    for j=1:size(new_tunings_array, 2)
        size(V,2)+1-l;
        g = scatter(transpose(V(:,(size(V,2)+1-l)))*new_tunings_array(l,j), l, 'filled');
        %g = scatter(transpose(V(:,(size(V,2)+1-l)))*new_tunings_array(l,j), l, colours(mod(j,size(colours,2))+1), 'filled');
        if (l==2)
            g_all = [g_all; g];
        end
        hold on
    end
end            
legend(legendStrings)

ylim([0 3])

% PCA clustered dataset


% clustering


% cur_T = size(trial(n,k).spikes(i,:),2);
% num_counts = sum(trial(:,k).spikes(start_t:cur_T+end_t),2);

% i = 1;
% tunings = zeros(K);
% for k=1:K
%     for n=1:N
%         total = sum(trial(n,k).spikes(i,:));
%         time = size(trial(n,k).spikes(i,:),2);
%         rate = total/time;
%     end
%     tunings(k) = avg_rate;
% end

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