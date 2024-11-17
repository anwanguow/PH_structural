f = trainedModel;
load('pca/pca_method_Exp_1_decay_p_98.mat');

correctness = 0.15;

dist(1) = 0;
for i = 0 : 80
    name = "data/X_t_" + string(100+10*i) + ".csv";
    X = readtable(name);

    test_features_centered = table2array(X) - pca_method.mu;
    test_score = test_features_centered * pca_method.coeff;
    new_test_features = array2table(test_score);
    X = new_test_features;
    columnNames = X.Properties.VariableNames;
    newColumnNames = strrep(columnNames, 'test_score', 'score');
    X.Properties.VariableNames = newColumnNames;

    [yfit,scores] = f.predictFcn(X);
    w = f.ClassificationSVM.Beta;
    norm_w = norm(w);
    s = scores(:, 2);
    distance = s / norm_w * f.ClassificationSVM.KernelParameters.Scale-correctness;
    dist(i+1) = mean(distance);
    if mod(i,10) == 0
        name
        i
    end
end

fname = "softness/softness.csv";
csvwrite(fname, dist);