f = trainedModel;
XTest = testset(:, 1:end-1);
YTest = testset(:, end);
YTest = YTest.Var1601;
YPredicted = f.predictFcn(XTest);
confMat = confusionmat(YTest, YPredicted);

% Calculate the Matthews correlation coefficient for multi-class
n = sum(confMat(:)); % Total number of samples
num_classes = size(confMat, 1);
sum1 = 0;
sum2 = 0;
sum3 = 0;

for k = 1:num_classes
    TP = confMat(k, k);
    FP = sum(confMat(:, k)) - TP;
    FN = sum(confMat(k, :)) - TP;
    TN = n - TP - FP - FN;
    sum1 = sum1 + TP * TN - FP * FN;
    sum2 = sum2 + (TP + FP) * (TP + FN);
    sum3 = sum3 + (TN + FP) * (TN + FN);
end

numerator = sum1;
denominator = sqrt(sum2) * sqrt(sum3);

if denominator == 0
    mcc = NaN;
else
    mcc = numerator / denominator;
end

disp('Confusion matrix:');
disp(confMat);
fprintf('Matthews Correlation Coefficient (MCC): %.3f\n', mcc);
