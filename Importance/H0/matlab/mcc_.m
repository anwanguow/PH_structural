f = trainedModel;
XTest = new_testset(:, 1:end-1);
YTest = new_testset(:, end);
YTest = YTest.Var1601;
YPredicted = f.predictFcn(XTest);
confMat = confusionmat(YTest, YPredicted);

TP = confMat(1, 1);
FN = confMat(1, 2);
FP = confMat(2, 1);
TN = confMat(2, 2);
numerator = (TP * TN) - (FP * FN);
denominator = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));

if denominator == 0
    mcc = NaN;
else
    mcc = numerator / denominator;
end

disp('Confusion matrix:');
disp(confMat);
fprintf('Matthews Correlation Coefficient (MCC): %.3f\n', mcc);
