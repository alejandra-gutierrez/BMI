%% Brain Machine Interfaces 2017/2018
%  Practice assignment - SVM classification

%% Initialization
clear ; close all; clc

%% =============== Part 1: Linear data ================

fprintf('Loading and Visualizing Data ...\n')

% Load from data1: 
% You will have X, y in your environment
load('data1.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ==================== Part 2: Training Linear SVM ====================

% Load from data1: 
% You will have X, y in your environment
load('data1.mat');

fprintf('\nTraining Linear SVM ...\n')

% You should try to change the C value below and see how the decision
% boundary varies (e.g., try C = 1000)
C = 1;
model = svmTrain(X, y, C, @linearKernel);
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =============== Part 3: Visualizing Dataset 2 ================

fprintf('Loading and Visualizing Data ...\n')

% Load from data2: 
% You will have X, y in your environment
load('data2.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Part 4: Training SVM with RBF Kernel (Dataset 2) ==========
%  After you have implemented the kernel, we can now use it to train the 
%  SVM classifier.
% 
fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');

% Load from ex6data2: 
% You will have X, y in your environment
load('data2.mat');

% SVM Parameters
C = 1; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice, you will want to run the training to
% convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 

visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Part 5: Third dataset ==========

fprintf('Loading and Visualizing Data ...\n')

% Load from data3: 
% You will have X, y in your environment
load('data3.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

% lots of options...