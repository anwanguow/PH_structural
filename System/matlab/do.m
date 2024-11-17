classificationLearner
D1 = readtable('dataset/H0H1/dataset_set_1.csv');
D2 = readtable('dataset/H0H1/dataset_set_2.csv');

features = D1(:, 1:end-1);
labels = D1(:, end);

varianceToKeep = 98;

[coeff, score, latent, tsquared, explained, mu] = pca(table2array(features));
cum_explained = cumsum(explained);
numComponents = find(cum_explained >= varianceToKeep, 1);
score = score(:, 1:numComponents);
coeff = coeff(:, 1:numComponents);
new_features = array2table(score);
trainset = [new_features labels];
pca_method.coeff = coeff;
pca_method.mu = mu;
pca_method.numComponents = numComponents;

% save('pca/pca_method_45.mat', 'pca_method');

test_features = D2(:, 1:end-1);
test_labels = D2(:, end);

test_features_centered = table2array(test_features) - pca_method.mu;
test_score = test_features_centered * pca_method.coeff;

new_test_features = array2table(test_score);
testset = [new_test_features test_labels];

trainset_colnames = trainset.Properties.VariableNames;
testset_colnames = testset.Properties.VariableNames;
testset.Properties.VariableNames = trainset_colnames;

classificationLearner


