%% Test input values
% PCA and linear regression
% go to plot_prediction to plot 

disp_dir =1 ;
disp_n = 1;

windowsize = 26;
t_mvt = 280;

run train_PCA_linReg.m

[handPos_estimated_x, handPos_estimated_y, velx_estimated, vely_estimated, errX, errY] = testModelPCA_function(trials_test, model, disp_dir, disp_n, windowsize, t_mvt);

run plot_prediction.m

save('estimations.mat','model', 'handPos_estimated_x', 'handPos_estimated_y', 'velx_estimated', 'vely_estimated', '-mat' );
