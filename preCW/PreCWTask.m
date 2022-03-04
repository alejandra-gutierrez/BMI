%% Section 1: Linear SVM Classifier

load('data1.mat')
% X is coordinates of data points
% y is corresponding label for each data point
plotData(X,y)
% data with one label plotted as dots, crossed for other label
C = 1;    % Misclassification penalty - adds more or less constraints on classification
% small C: prioritises correct classification of large groups of points but
% can lead to misclassification of single points
% large C: prioritises correct classficiation of all points without
% exception 
model = svmTrain(X, y, C, @linearKernel);
visualizeBoundary(X,y,model);

% With C = 1.0, 100% correct classification occurs - large distance between
% two distinct groups 

% for this data set, 100% classification success occurs for all(?) values
% of C. Shifts positioning of line - looks pretty central at c = 1

%% Section 2: Non-Linear SVM Classifier

% RBF kernal allows to measure similarity in space between datapoints - the
% closer datapoints are in 2D space gives bigger value of RBF function 

load('data2.mat')
% plotData(X, y)
C = 100;    % misclassification penalty as above
sigma = 0.03;    % parameter of RBF ( radial basis function) kernel function describing its spread
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); % creates data structure containing parameter of trained RBF-based SVM classifier 
visualizeBoundary(X, y, model);

% C = 1.0, sigma = 0.1 - not 100% accuracy 
% C = 10, sigma = 0.055 - better accuracy - close to 100% 
% 100% accuracy when C = 100, sigma = 0.03

% smaller sigma does better at finding 

%% Section 3: data3.mat 
load('data3.mat')
C = 2;
sigma = 0.085;
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
figure
visualizeBoundary(X, y, model);

model = svmTrain(X, y, C, @linearKernel);
figure
visualizeBoundary(X,y,model);

model = svmTrain(Xval, yval, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
figure
visualizeBoundary(Xval, yval, model);

model = svmTrain(Xval, yval, C, @linearKernel);
figure
visualizeBoundary(Xval,yval,model);
