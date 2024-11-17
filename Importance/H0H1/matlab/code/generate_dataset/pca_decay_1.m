load('dataset/exp_1/Exp_1_decay_tau_0.1.mat');
features = trainset(:, 1:end-1);
labels = trainset(:, end);

% 设置保留的方差百分比
varianceToKeep = 99.9999999999999;

% 对特征矩阵进行PCA
[coeff, score, latent, tsquared, explained, mu] = pca(table2array(features));

% 计算需要的主成分数以保留指定百分比的方差
cum_explained = cumsum(explained);
numComponents = find(cum_explained >= varianceToKeep, 1);

% 仅保留指定数量的主成分
score = score(:, 1:numComponents);
coeff = coeff(:, 1:numComponents);

% 创建新的table，包含PCA后的特征和原始分类标签
new_features = array2table(score);
new_trainset = [new_features labels];

% 保存PCA的方法
pca_method.coeff = coeff;
pca_method.mu = mu;
pca_method.numComponents = numComponents;

% 如果需要保存到文件，可以使用如下命令：
% save('pca/pca_method_Exp_1_decay_tau_0.1_p_50.mat', 'pca_method');


% 假设testset已经在工作区中

% 提取testset的特征矩阵（前n-1列）和分类标签（最后一列）
test_features = testset(:, 1:end-1);
test_labels = testset(:, end);

% 使用保存的PCA方法对testset进行PCA
test_features_centered = table2array(test_features) - pca_method.mu;
test_score = test_features_centered * pca_method.coeff;

% 创建新的table，包含PCA后的特征和原始分类标签
new_test_features = array2table(test_score);
new_testset = [new_test_features test_labels];