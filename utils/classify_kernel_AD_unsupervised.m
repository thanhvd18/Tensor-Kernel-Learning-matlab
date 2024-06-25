function [C, acc] = classify_kernel_AD_unsupervised(K,Y,train_index,test_index)

y_train = Y(train_index); 
y_test = Y(test_index); 

K_train = K(train_index,train_index);
K_test = K(test_index,train_index);

numTrain = length(train_index);
numTest =  length(test_index);


K_train = [(1:numTrain)' ,  K_train];
K_test  = [(1:numTest)' ,   K_test];


model = svmtrain(y_train, K_train, '-t 4 -b 1');
[predicted_label, accuracy, decision_values] = svmpredict(y_train, K_train, model, '-b 1');
C = confusionmat(y_train,predicted_label);
fprintf('========= Train acc: %g =========\n',accuracy(1) )
[predicted_label, accuracy, decision_values] = svmpredict(y_test, K_test, model, '-b 1');
C = confusionmat(y_test,predicted_label);
fprintf('========= Test acc: %g =========\n',accuracy(1) )
acc = accuracy(1);
end
