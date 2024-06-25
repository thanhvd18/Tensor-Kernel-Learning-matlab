function [accuracy, specificity, sensitivity] = computeMetrics(C)
    % ComputeMetrics Calculates accuracy, specificity, and sensitivity from a confusion matrix
    %   C: A 2x2 confusion matrix
    %       C = [TP, FN;
    %            FP, TN]
    %   accuracy: Overall, how often is the classifier correct?
    %   specificity: When it's actually no, how often does it predict no?
    %   sensitivity: When it's actually yes, how often does it predict yes?

    TP = C(1,1);
    FN = C(1,2);
    FP = C(2,1);
    TN = C(2,2);
    
    accuracy = (TP + TN) / sum(C(:)); % Calculate accuracy
    specificity = TN / (TN + FP); % Calculate specificity
    sensitivity = TP / (TP + FN); % Calculate sensitivity
end