load('dataset/exp_1/Exp_1_decay.mat');
features = trainset(:, 1:end-1);
labels = trainset(:, end);

varianceToKeep = 40;

[coeff, score, latent, tsquared, explained, mu] = pca(table2array(features));

cum_explained = cumsum(explained);
numComponents = find(cum_explained >= varianceToKeep, 1);

score = score(:, 1:numComponents);
coeff = coeff(:, 1:numComponents);

new_features = array2table(score);
new_trainset = [new_features labels];

pca_method.coeff = coeff;
pca_method.mu = mu;
pca_method.numComponents = numComponents;

save('pca/pca_method_Exp_1_decay_p_40.mat', 'pca_method');

test_features = testset(:, 1:end-1);
test_labels = testset(:, end);
test_features_centered = table2array(test_features) - pca_method.mu;
test_score = test_features_centered * pca_method.coeff;

new_test_features = array2table(test_score);
new_testset = [new_test_features test_labels];

trainset_colnames = new_trainset.Properties.VariableNames;
testset_colnames = new_testset.Properties.VariableNames;
new_testset.Properties.VariableNames = trainset_colnames;
