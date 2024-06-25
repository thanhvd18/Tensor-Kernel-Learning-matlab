function [X_train, X_test, y_train, y_test] = train_test_split(X,y,test_size, random_state)
rand('seed', random_state);
N = length(X);
idx = randperm(N);
X_train = X(idx(1:round(N*(1-test_size))), :);
X_test = X(idx(1:round(N*(test_size))),:);

y_train = y(idx(1:round(N*(1-test_size))));
y_test = y(idx(1:round(N*(test_size))));