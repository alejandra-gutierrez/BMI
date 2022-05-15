function [modelParameters] = positionEstimatorTraining(training_data)
% Arguments:

% - training_data:
%     training_data(n,k)              (n = trial id,  k = reaching angle)
%     training_data(n,k).trialId      unique number of the trial
%     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
%     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)

% ... train your model

% Return Value:

% - modelParameters:
%     single structure containing all the learned parameters of your
%     model and which can be used by the "positionEstimator" function.

t_start = tic;
tic;
[N_trials_tr, N_angles] = size(training_data);
N_neurons = size(training_data(1).spikes, 1);

windowsize = 15;
t_mvt = 210;
t_pre_mvt = 300;
t_step = windowsize/2;
t_step = ceil(t_step);
n_neighbours = 12;
proportion = 2/100; % th for selection of principal components

fprintf("\nFinding spike rates and velocities...");
[velx_tr, vely_tr, ~] = getvel2(training_data, windowsize, t_step, t_mvt); % one array with all three axis 
% pos_tr = get_pos(training_data, windowsize, t_step, t_mvt);
% acc_tr = getacc(training_data, windowsize, t_step, t_mvt);
spike_rate = get_spike_rates2(training_data, windowsize, t_step, t_mvt);
fprintf("Spike_Rate done...\n");
toc;

%% TRAIN KNN MODEL

fprintf("Training KNN model...");
spikesr = zeros(N_angles*N_trials_tr, N_neurons);
labels = zeros(1, N_angles*N_trials_tr);
for k_it = 1:N_angles
    for n_it = 1:N_trials_tr
         spikesr( (k_it-1)*N_trials_tr + n_it, :) = sum(training_data(n_it, k_it).spikes(:, 1:t_pre_mvt), 2)';           
        labels( (k_it-1)*N_trials_tr + n_it) = k_it;
    end
end

% knn = fitcknn(spikesr, labels);
for k_it = 1:N_angles+1
    modelParameters(k_it).KNNSpikesr = spikesr;
    modelParameters(k_it).KNNLabels = labels;
    % modelParameters(k_it).knn = knn;
    modelParameters(k_it).n_neighbours = n_neighbours;
end
fprintf("KNN model done. "); toc;


%% TRAIN POSITION ESTIMATOR
fprintf("Extracting Principal component vectors from data...");
%   spike_rate_av_trials = make_av_spike_rate(spike_rate);
%   [principal_spikes, Vs, Ds, M] = spikes_PCA(spike_rate_av_trials, 0.05);
%   principal_spikes_tr = cell(N_trials_tr, N_angles+1);
% 
%   modelParameters(9).M = M;
  modelParameters(9).dir = 0;
%   modelParameters(9).Vs = Vs;
%   modelParameters(9).Ds = Ds;
%   modelParameters(9).V_red = Vs(:,1:M);


for k_it =0:N_angles
    spike_rate_av_trials = make_av_spike_rate(spike_rate, k_it);
    [~, Vs, Ds, M] = spikes_PCA(spike_rate_av_trials, proportion);
    dir = k_it;
    if k_it == 0
        k_it = N_angles+1;
    end
    V_red = Vs(:, 1:M);
    modelParameters(k_it).M = M; % keep same M for all
    modelParameters(k_it).dir = dir;
    modelParameters(k_it).Vs = Vs;
    modelParameters(k_it).Ds = Ds;
    modelParameters(k_it).V_red = V_red;
    modelParameters(k_it).MdlnetX = [];
    modelParameters(k_it).MdlnetY = [];

    for n_it = 1:N_trials_tr
        if k_it == N_angles+1
            for k = 1:N_angles
                % make a specific array for non-specific training
                spikes_mean = mean(spike_rate{n_it, k}, 2);
                principal_spikes_0{n_it, k} = V_red'*(spike_rate{n_it, k}-spikes_mean);
            end
        else
           spikes_mean = mean(spike_rate{n_it, k_it}, 2);
           principal_spikes_tr{n_it, k_it} = V_red'*(spike_rate{n_it, k_it}-spikes_mean);
        end
        fprintf(".");
    end
    fprintf("\n"); 
end
fprintf("Extracted PCA parameters.\n"); toc;

fprintf("Starting Neural Networks Training.\t");
% x_results_dir = cell(1, 9);
% y_results_dir = cell(1, 9);

for k_it = 0:N_angles
    fprintf("k=%g.\t", k_it);
    if (k_it ==0) % non-direction specific training
        [input_datax, output_datax] = linearizeInputOutput(principal_spikes_0, velx_tr, k_it);
        [input_datay, output_datay] = linearizeInputOutput(principal_spikes_0, vely_tr, k_it);

%         [A, W, H, Q]= kalmanCoeffs(input_data, output_data);

% ---------- CODE TO FIND MODEL AND PARAMETERS ------------ % 
        [~, x_results] = fitrauto(input_datax', output_datax,"Learners", ["gp", "net", "linear"],"HyperparameterOptimizationOptions",struct("Optimizer","asha","MaxTime", 600));
        disp("x results, k = 0 done")
        [~, y_results] = fitrauto(input_datay', output_datay,"Learners", ["gp", "net", "linear"],"HyperparameterOptimizationOptions",struct("Optimizer","asha","MaxTime", 600));
        disp("y results, k = 0 done")

       
        x_results_dir{1, 9} = x_results;
        y_results_dir{1, 9} = y_results;
% --------------------------------------------------------- % 

% % ------------------ 11.05.2022 Test 1 -------------------- % 
%         x_activation = 'none';
%         y_activation = 'none';
%         x_lambda = 1.35313412818168e-08;
%         y_lambda = 1.79753303368664e-09;
%         x_standardize = false;
%         y_standardize = false;
%         x_layers = [3 5];
%         y_layers = [8];
% % --------------------------------------------------------- % 

% % ------------------ 11.05.2022 Test 2 -------------------- % 
%         x_activation = 'none';
%         y_activation = 'none';
%         x_lambda = 2.26216301188866e-10;
%         y_lambda = 1.84010252449154e-10;
%         x_standardize = false;
%         y_standardize = false;
%         x_layers = [1 4 170];
%         y_layers = [1 33];
% % --------------------------------------------------------- % 

%         mdl_x = fitrnet(input_datax, output_datax,'Activations', x_activation, 'Lambda', x_lambda, 'Standardize', x_standardize, 'LayerSizes', x_layers, 'ObservationsIn','columns');
%         mdl_y = fitrnet(input_datay, output_datay,'Activations', y_activation, 'Lambda', y_lambda, 'Standardize', y_standardize, 'LayerSizes', y_layers, 'ObservationsIn','columns');
        

        k_it = N_angles+1;
    else  % direction specific training
        [input_datax, output_datax] = linearizeInputOutput(principal_spikes_tr, velx_tr, k_it);
        [input_datay, output_datay] = linearizeInputOutput(principal_spikes_tr, vely_tr, k_it);


% ---------- CODE TO FIND MODEL AND PARAMETERS ------------ % 
        [~, x_results] = fitrauto(input_datax', output_datax,"Learners", ["gp", "net", "linear"],"HyperparameterOptimizationOptions",struct("Optimizer","asha","MaxTime", 600));
        disp("x results, k = "+k_it+" done")
        [~, y_results] = fitrauto(input_datay', output_datay, "Learners", ["gp", "net", "linear"],"HyperparameterOptimizationOptions",struct("Optimizer","asha","MaxTime", 600));
        disp("y results, k = "+k_it+" done")

        x_results_dir{1, k_it} = x_results;
        y_results_dir{1, k_it} = y_results;
% --------------------------------------------------------- % 

% % ------------------ 11.05.2022 Test 1 -------------------- % 
% 
%         % 20220511_test_1: overlapping windows, removal of factor of 2 in getvel2.m
%         switch k_it
%             case 1 
%                 x_activation = 'sigmoid';
%                 y_activation = 'sigmoid';
%                 x_lambda = 5.26126101367222e-09;
%                 y_lambda = 2.39900825370355e-10;
%                 x_standardize = false;
%                 y_standardize = false;
%                 x_layers = [158];
%                 y_layers = [122 10];
%             case 2
%                 x_activation = 'relu';
%                 y_activation = 'sigmoid';
%                 x_lambda = 7.38069993914245e-06;
%                 y_lambda = 0.000250739701973964;
%                 x_standardize = true;
%                 y_standardize = false;
%                 x_layers = [1];
%                 y_layers = [7 2];
%             case 3
%                 x_activation = 'tanh';
%                 y_activation = 'tanh';
%                 x_lambda = 0.01962600028231;
%                 y_lambda = 0.00594350936327051;
%                 x_standardize = false;
%                 y_standardize = false;
%                 x_layers = [8];
%                 y_layers = [4 2 1];
%             case 4 
%                 x_activation = 'relu';
%                 y_activation = 'tanh';
%                 x_lambda = 0.0263093692996972;
%                 y_lambda = 3.67759968227727e-09;
%                 x_standardize = true;
%                 y_standardize = false;
%                 x_layers = [34];
%                 y_layers = [1 1];
%             case 5
%                 x_activation = 'sigmoid';
%                 y_activation = 'none';
%                 x_lambda = 0.000268657518635669;
%                 y_lambda = 4.67539961000468e-10;
%                 x_standardize = true;
%                 y_standardize = true;
%                 x_layers = [4];
%                 y_layers = [31];          
%             case 6
%                 x_activation = 'sigmoid';
%                 y_activation = 'tanh';
%                 x_lambda = 8.20101480235253e-08;
%                 y_lambda = 0.0000011217772702795;
%                 x_standardize = false;
%                 y_standardize = true;
%                 x_layers = [15 6];
%                 y_layers = [2];         
%             case 7
%                 x_activation = 'sigmoid';
%                 y_activation = 'tanh';
%                 x_lambda = 7.98627128202956e-07;
%                 y_lambda = 7.79798493638216e-07;
%                 x_standardize = true;
%                 y_standardize = true;
%                 x_layers = [1 2];
%                 y_layers = [1];            
%             case 8
%                 x_activation = 'tanh';
%                 y_activation = 'none';
%                 x_lambda = 5.20348980151386e-06;
%                 y_lambda = 7.36951785331708e-07;
%                 x_standardize = false;
%                 y_standardize = true;
%                 x_layers = [3];
%                 y_layers = [14 37];
%         end
% % --------------------------------------------------------- % 

% % ------------------ 11.05.2022 Test 2 -------------------- % 
% 
%         % 20220511_test_2: non overlapping windows, removal of factor of 2 in getvel2.m
%         switch k_it
%             case 1 
%                 x_activation = 'tanh';
%                 y_activation = 'none';
%                 x_lambda = 0.0000942272116847018;
%                 y_lambda = 0.00128472775650572;
%                 x_standardize = false;
%                 y_standardize = false;
%                 x_layers = [1];
%                 y_layers = [1];
%             case 2
%                 x_activation = 'none';
%                 y_activation = 'relu';
%                 x_lambda = 1.43337640569499E-10;
%                 y_lambda = 0.00346132340253797;
%                 x_standardize = true;
%                 y_standardize = false;
%                 x_layers = [14];
%                 y_layers = [4 48];
%             case 3
%                 x_activation = 'none';
%                 y_activation = 'none';
%                 x_lambda = 5.08428822263802E-06;
%                 y_lambda = 0.00538886165573154;
%                 x_standardize = false;
%                 y_standardize = false;
%                 x_layers = [5];
%                 y_layers = [1];
%             case 4 
%                 x_activation = 'tanh';
%                 y_activation = 'sigmoid';
%                 x_lambda = 0.000083031624345349;
%                 y_lambda = 0.0000628000050651263;
%                 x_standardize = false;
%                 y_standardize = true;
%                 x_layers = [7 1 2];
%                 y_layers = [1];
%             case 5
%                 x_activation = 'tanh';
%                 y_activation = 'sigmoid';
%                 x_lambda = 0.000351472700964922;
%                 y_lambda = 8.30640481876404E-06;
%                 x_standardize = false;
%                 y_standardize = true;
%                 x_layers = [4 1];
%                 y_layers = [1];          
%             case 6
%                 x_activation = 'tanh';
%                 y_activation = 'tanh';
%                 x_lambda = 0.0151288459968471;
%                 y_lambda = 0.000140169854149635;
%                 x_standardize = false;
%                 y_standardize = false;
%                 x_layers = [8];
%                 y_layers = [2];         
%             case 7
%                 x_activation = 'tanh';
%                 y_activation = 'sigmoid';
%                 x_lambda = 0.000216090599991575;
%                 y_lambda =0.000820791821519997;
%                 x_standardize = false;
%                 y_standardize = false;
%                 x_layers = [79 1 2];
%                 y_layers = [6];            
%             case 8
%                 x_activation = 'relu';
%                 y_activation = 'none';
%                 x_lambda = 0.0311012435072188;
%                 y_lambda = 0.00129477280992996;
%                 x_standardize = false;
%                 y_standardize = true;
%                 x_layers = [10 172];
%                 y_layers = [2 3];
%         end
% % --------------------------------------------------------- % 

%         [A, W, H, Q]= kalmanCoeffs(input_data, output_data);
%         mdl_x = fitrnet(input_datax, output_datax,'Activations', x_activation, 'Lambda', x_lambda, 'Standardize', x_standardize, 'LayerSizes', x_layers, 'ObservationsIn','columns');
%         mdl_y = fitrnet(input_datay, output_datay,'Activations', y_activation, 'Lambda', y_lambda, 'Standardize', y_standardize, 'LayerSizes', y_layers, 'ObservationsIn','columns');



    end

%     modelParameters(k_it).x = mdl_x;
%     modelParameters(k_it).y = mdl_y;
%     modelParameters(k_it).A = A;
%     modelParameters(k_it).W = W;
%     modelParameters(k_it).H = H;
%     modelParameters(k_it).Q = Q;
%     modelParameters(k_it).P = zeros(size(A,1)); % initialise a priori covariance matrix
    

%     modelParameters(k_it).MdlnetX = MdlnetX;
%     modelParameters(k_it).MdlnetY = MdlnetY;

end
fprintf("\n Done.\n");
fprintf("Model Parameters:\n");
% print model parameters
for k_it = 1:N_angles+1
%     M = modelParameters(k_it).M;
%     dir = modelParameters(k_it).dir;
    V_red = modelParameters(k_it).V_red;
%     Vs = modelParameters(k_it).Vs;
%     Ds = modelParameters(k_it).Ds;
%     fprintf("dir=%g, M=%g,  size V_red=[%g, %g], size wX=[%g,%g], size wY=[%g,%g]\n",...
%     dir, M, size(V_red,1),size(V_red,2), size(wX,1), size(wX, 2), size(wY,1), size(wY, 2));
end
fprintf("\nFinished Training.\n");
toc; fprintf("\n");
end
