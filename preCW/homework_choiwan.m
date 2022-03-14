%% Initialisation
% for tuning curves only, run this first section, the run the final section
% (5)

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

%% 1) population raster plot, 1 trial, multi neural units
% 'a spike train of duration T ms is represented by a 1*T vector.'
% 'spike train recorded from i-th unit of n-th trial of k-th reaching
% angle'
% trial(n,k).spikes(i,:)
% i=1:98, n=1:100, k=1:8, t=1:672

n = 1;
k = 1;
spikes = transpose(trial(n,k).spikes(1:2,:));
figure
for i=1:I
    hold on
    scatter(1:T, trial(n,k).spikes(i,:)*i, 'o', 'filled')
end
ylim([1 I])
xlabel('Time / bins')
ylabel('Neural units')
title('Population raster plot for one trial')

%% 2) population raster plot, multi trials, 1 neural unit
% needs padding of zeros

% k = 1;
% i = 1;
% figure
% a = zeros(100);
% for j=1:100
%     a(j) = size(trial(j,k).spikes(i,:),2);
% end
% m = min(a)
% for n=1:N
%     hold on
%     n
%     T
%     size(trial(n,k).spikes(i,:))
%     scatter(1:T, trial(n,k).spikes(i,1:600));
% end
% xlabel('Time / bins')
% ylabel('Neural units')
% title('Population raster plot for one neural units over many trials')

%% 3) PSTH for different neural units

%% one trial

n = 1;
k = 1;
count = zeros(T);
for t=1:T
    for i=1:I
        if (trial(n,k).spikes(i,t)==1)
            count(t) = count(t)+1;
        end
    end
end
figure
plot(1:T, count)
% xlim([300 572])
xlabel('Time / bins')
ylabel('Firing rate')
title('PSTH, one trial')

%% averaged across trials
% uses smoothfilter() with Gaussian filtering

k = 8;
i = 10;
total_spikes = zeros(1,N);
T_max = 0;
T_temp = 0;

% find number of zeros to pad
for k=1:8
    for n=1:N
        T_temp = size(trial(n,k).spikes(i,:),2);
        T_max = max(T_temp, T_max);
    end
end
spikes_padded = zeros(N, T_max);

% pad every dataset with zeros
for k=1:8
    for n=1:N
        spikes_cur = trial(n,k).spikes(i,:);
        T_cur = size(spikes_cur,2);
        if (T_cur<T_max)
            spikes_padded(n,:) = cat(2, spikes_cur, zeros([1 T_max-T_cur]));
        end
    end
end

% plot PSTH
total_spikes = sum(spikes_padded);
rates = total_spikes/T_max;
rates_smooth = smoothdata(rates,'gaussian', 100);
% subplot(2,1,1)
% plot(1:T_max, rates)
% subplot(2,1,2)
% plot(1:T_max, rates_smooth)
figure
subplot(2,1,1)
plot(300:T_max-100, rates(300:T_max-100))
subplot(2,1,2)
plot(300:T_max-100, rates_smooth(300:T_max-100))
title('PSTH, all trials')

%% 4) Hand positions for different trials
% Plots x,y,z coordinate projections individually

% arm movement
% handPos(1/2/3,:)

k = 1;
figure
for n=1:8
    subplot(4,2,n)
    for d=1:3
        hold on
        T_unique = size(trial(n,k).handPos(d,:),2);
        plot(1:T_unique, trial(n,k).handPos(d,:))
    end
    xlabel('Time / bins')
    ylabel('Hand positions')
    legend('1', '2', '3')
end

% Cross-trial results generally follow similar trajectories, whilst varying
% slightly in shape.

%% 5) Tuning curves for several neurons
close all

start_t = 300;
end_t = -100;

legend_array = [];
tunings_array = [];
figure

colours = ['b','r','g','y','m','c','k'];
l = 1;
for i=1:98
    tunings = p2t(trial,i);
    %tunings_all(i,:) = tunings;

    % threshold - comment out to plot all 98 neurons
    range = max(tunings) - min(tunings);
%     if (range > 0.005) && (min(tunings) > 0.02)
%         legend_array = [legend_array; i];
%         tunings_array = [tunings_array; tunings];
%         plot(1:K, tunings,colours(l),'DisplayName', num2str(i));
%         l = l+1;
%         hold on
%     end
    legend_array = [legend_array; i];
    tunings_array = [tunings_array; tunings];
    plot(1:K, tunings,'DisplayName', num2str(i));
    hold on
end
legendStrings = "i = " + string(legend_array);
legend(legendStrings)
title('All tuning curves')

tunings_array = normalize(tunings_array);
C = cov(tunings_array);
[V, D] = eig(C);
last = size(V,2);
F = [V(:,last) V(:,last-1)];

% eigenvalues
figure
plot(8:-1:1, max(D)')

% 2D PCA plot
new_tunings_array = transpose(F) * transpose(tunings_array);
figure
scatter(new_tunings_array(1,:), new_tunings_array(2,:))
hold on

new_C = cov(transpose(new_tunings_array))
[new_V,new_D] = eig(new_C)
new_V = -new_V;
new_D = -new_D;
new_C = -new_C;
for j=1:2
    plot([0 new_V(1,j)], [0 new_V(2,j)], 'LineWidth', 2)
    hold on
end

g_all = [];
% 1D PCA plot
figure
for l=1:2
    for j=1:size(new_tunings_array, 2)
        size(V,2)+1-l
        g = scatter(transpose(V(:,(size(V,2)+1-l)))*new_tunings_array(l,j), l, colours(j), 'filled');
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