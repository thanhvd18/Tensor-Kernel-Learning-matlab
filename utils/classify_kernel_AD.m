function [C, acc] = classify_kernel_AD(K,K_y,Z)

Y = squeeze(K_y(1,1,:));
Y = double(categorical(Y));

%%
train_index = [Z{1} Z{2}] +1; %train + val, rescale 0-(N-1) -> 1-N
test_index = Z{3} +1;
train_index = double(train_index);
test_index = double(test_index);

K_train = K(train_index,train_index);
y_train = Y(train_index);

K_test = K(test_index,train_index);
y_test = Y(test_index);

numTrain = length(train_index);
numTest =  length(test_index);


K_train = [(1:numTrain)' ,  K_train];
K_test  = [(1:numTest)' ,   K_test];

acc = 0;
model = svmtrain(y_train, K_train, '-t 4 -b 1');
[predicted_label, accuracy_train, decision_values] = svmpredict(y_train, K_train, model, '-b 1');
C = confusionmat(y_train,predicted_label)
[predicted_label, accuracy, decision_values] = svmpredict(y_test, K_test, model, '-b 1');
C = confusionmat(y_test,predicted_label);
fprintf('========= Train acc: %g =========\n',accuracy_train(1) )
fprintf('========= Test acc: %g =========\n',accuracy(1) )
acc = accuracy(1);
end
