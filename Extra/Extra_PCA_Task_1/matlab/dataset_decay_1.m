tau = 0.57;

sample = [];
for i = 0:9
    for j = 1:9
        D = readtable('database/decay/set_1/D_' + string(i) + '/X_t_' + string(j*100) + '_tau_' + string(tau) + '.csv');
        sample = [sample; D];
    end
end
labels = sample(:, end);
labels = table2array(labels);

m_1 = 7500;
n_1 = 7500;
idx_label_1 = find(labels == 1);
rand_idx_label_1 = randsample(idx_label_1, 15000);
set_1 = sample(rand_idx_label_1, :);
numRows_1 = size(set_1, 1);
randomIndices_1 = randperm(numRows_1);
train_1 = set_1(randomIndices_1(1:m_1), :);
test_1 = set_1(randomIndices_1(end-n_1+1:end), :);

m_0 = 7500;
n_0 = 7500;
idx_label_0 = find(labels == 0);
rand_idx_label_0 = randsample(idx_label_0, 15000);
set_0 = sample(rand_idx_label_0, :);
numRows_0 = size(set_0, 1);
randomIndices_0 = randperm(numRows_0);
train_0 = set_0(randomIndices_0(1:m_0), :);
test_0 = set_0(randomIndices_0(end-n_0+1:end), :);

trainset = vertcat(train_1, train_0);
testset = vertcat(test_1, test_0);
rank_train = randperm(size(trainset, 1));
rank_test = randperm(size(testset, 1));

trainset = trainset(rank_train,:);
testset = testset(rank_test,:);


