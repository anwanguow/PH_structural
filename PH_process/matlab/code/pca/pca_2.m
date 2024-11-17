load('dataset/exp_2/Exp_2_decay.mat');

load('pca/pca_method_Exp_1_decay_p_40.mat');

test_features = testset(:, 1:end-1);
test_labels = testset(:, end);

test_features_centered = table2array(test_features) - pca_method.mu;
test_score = test_features_centered * pca_method.coeff;
new_test_features = array2table(test_score);
new_testset = [new_test_features test_labels];

colNames = new_testset.Properties.VariableNames;

for i = 1:length(colNames)-1
    colNames{i} = strrep(colNames{i}, 'test_score', 'score');
end

new_testset.Properties.VariableNames(1:end-1) = colNames(1:end-1);
