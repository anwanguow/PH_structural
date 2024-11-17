% 读取数据集
D1 = readtable('dataset/H0H1H2/dataset_set_1.csv');

% 分离特征和标签
features = D1(:, 1:end-1);
labels = D1(:, end);

% 设置要保留的方差比例
varianceToKeep = 45;

% 进行PCA分析
[coeff, score, latent, tsquared, explained, mu] = pca(table2array(features));

% 计算累积解释方差并确定保留的主成分数量
cum_explained = cumsum(explained);
numComponents = find(cum_explained >= varianceToKeep, 1);

% 保留前 numComponents 个主成分
score = score(:, 1:numComponents);
coeff = coeff(:, 1:numComponents);
new_features = array2table(score);
trainset = [new_features labels];
pca_method.coeff = coeff;
pca_method.mu = mu;
pca_method.numComponents = numComponents;

% 计算每个原始变量在所有主成分上的绝对系数和
absolute_coeff_sum = sum(abs(coeff), 2);

% 找到绝对系数和最大的变量索引
[~, max_index] = max(absolute_coeff_sum);

% 输出最大权重变量的索引
disp(['The most important variable index is: ', num2str(max_index)]);
