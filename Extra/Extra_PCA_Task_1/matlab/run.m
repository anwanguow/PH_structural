% percentage - num of predictors
% 48% - 6
% 45% - 5
% 40% - 4
% 35% - 3
% 30% - 2
% 23% - 1

% Here to set the percentage of var explained in PCA
percentage_ = 48;

load('dataset/exp_2/Exp_2_decay_p_' + string(percentage_) + '.mat');
saved_session = "results/exp_2/Exp_1_p_" + string(percentage_) + ".mat";
classificationLearner(saved_session)

