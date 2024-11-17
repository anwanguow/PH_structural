load('dataset/exp_2/Exp_2_decay.mat');

load('pca/pca_method_Exp_1_decay_p_98.mat');

test_features = testset(:, 1:end-1);
test_labels = testset(:, end);

% 使用保存的PCA方法对testset进行PCA
test_features_centered = table2array(test_features) - pca_method.mu;
test_score = test_features_centered * pca_method.coeff;

% 创建新的table，包含PCA后的特征和原始分类标签
new_test_features = array2table(test_score);
new_testset = [new_test_features test_labels];

% 假设table的名字是T
% 获取table的列名
colNames = new_testset.Properties.VariableNames;

% 循环处理前n-1列
for i = 1:length(colNames)-1
    % 替换列名中的'test_score_'为'score_'
    colNames{i} = strrep(colNames{i}, 'test_score', 'score');
end

% 将修改后的列名重新赋值给table
new_testset.Properties.VariableNames(1:end-1) = colNames(1:end-1);
