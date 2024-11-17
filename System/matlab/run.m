classificationLearner('results/H0H1H2_80.mat');
D2 = readtable('dataset/H0H1H2/dataset_set_2.csv');
load('pca/pca_method_80.mat');


test_features = D2(:, 1:end-1);
test_labels = D2(:, end);

test_features_centered = table2array(test_features) - pca_method.mu;
test_score = test_features_centered * pca_method.coeff;

new_test_features = array2table(test_score);
testset = [new_test_features test_labels];

columnNames = testset.Properties.VariableNames;
newColumnNames = erase(columnNames, "test_");

testset.Properties.VariableNames = newColumnNames;
