clear;close all;clc;
run('init.m')
set(0,'DefaultFigureColormap',feval('hot'));
for exp=1:1
exp=2
load(sprintf("./ADvsCN/K_X_%i.mat", exp))
[M, N, ~] = size(K_X([1,2,3,4],:,:));
for ii = 1 : M
    X_ten(:,:,ii)  = K_X(ii,:,:);
end
X_ten = double(X_ten);
X_ten = tensor(X_ten);
for ii = 1 : M
    X_ten(:,:,ii)  = K_X(ii,:,:);
    K_temp = double(X_ten(:,:,ii));
    K_temp(1:N+1:end) = ones(N,1);
    X_ten(:,:,ii) = K_temp;
end
X_ten = double(X_ten);
X_ten = tensor(X_ten);

%% test with different rank
%% R = 2 % the best result 
R = 2;
[Result,a,b] = cp_als(X_ten,R,'tol', 1e-9, 'maxiters', 1000);
% Result =  tucker_als(X_ten,[R R R],'tol', 1e-9, 'maxiters', 1000); 

% opts.mu = 1e-9 ;
% opts.rho = 1.1;
% opts.max_iter = 500;
% opts.DEBUG = 1;
%  lambda = 1/sqrt(3*231);
% [Lhat,Shat] =  trpca_tnn(double(X_ten),lambda,opts);
% Result = Shat;

%% Cross diffusion process

% https://www.sciencedirect.com/science/article/pii/S0031320316303247
kNN = 3; %3
% iter = 3;
iter = 5;% 91.49
X_difusion = cross_diffusion_process(double(X_ten),kNN,iter);
for ii = 1 : M
    K_temp = X_difusion(:,:,ii);
    K_temp(1:N+1:end) = ones(N,1);
    X_difusion(:,:,ii) = K_temp;
end

%% SimpleMKL
Result = double(Result);
Y = squeeze(K_y(1,1,:));
Y = double(categorical(Y))*2-3;
for ii = 1 : M
    K_temp = Result(:,:,ii);
    K_temp(1:N+1:end) = ones(N,1);
    Result(:,:,ii) = K_temp;
end

[w1, K_combine,acc_test1] = MKL_wrap(double(X_ten),Y,Z);
[C, acc1] = classify_kernel_AD(K_combine,K_y,Z)
[w2, K_combine,acc_test2] = MKL_wrap(double(Result),Y,Z);
[C, acc2] = classify_kernel_AD(K_combine,K_y,Z)



%%
X_reconstructed = cross_diffusion_process(double(Result),kNN,iter);
X_reconstructed = double(full(X_reconstructed));


% %X_reconstructed = cross_diffusion_process(double(Result),kNN,iter);
% X_reconstructed = double(full(Result));
for ii = 1 : M
    K_temp = X_reconstructed(:,:,ii);
    K_temp(1:N+1:end) = ones(N,1);
    X_reconstructed(:,:,ii) = K_temp;
end
results = [];
for modality=1:M
K = squeeze(double(X_ten(:,:,modality)));
[Confusion, acc] = classify_kernel_AD(K,K_y,Z);
[accuracy, specificity, sensitivity] = computeMetrics(Confusion);

K = squeeze(X_difusion(:,:,modality));
[Confusion_diffusion,acc_diffusion] = classify_kernel_AD(K,K_y,Z);
[accuracy_diffusion, specificity_diffusion, sensitivity_diffusion] = computeMetrics(Confusion_diffusion);

K = squeeze(X_reconstructed(:,:,modality));
[Confusion_tensor,acc_tensor] = classify_kernel_AD(K,K_y,Z);
[accuracy_tensor, specificity_tensor, sensitivity_tensor] = computeMetrics(Confusion_tensor);
% acc_tensor_list(end+1) = acc_tensor;

% results(end+1,:) = [computeMetrics(Confusion),
%     computeMetrics(Confusion_diffusion),
%     computeMetrics(Confusion_tensor)];
results(end+1,:) = [accuracy,accuracy_diffusion,accuracy_tensor,accuracy_diffusion, specificity_diffusion, sensitivity_diffusion,accuracy_tensor, specificity_tensor, sensitivity_tensor];
end

%% Visualize results 
% X_reconstructed = double(full(Result));
% for i=1:M
%     X_reconstructed(:,:,i) = minMaxNormalize(X_reconstructed(:,:,i));
% end
for i=1:M
f= figure(i);
modality = i;
subplot(1,3,1)
imagesc(double(X_ten(:,:,i)))
xticklabels([])
yticklabels([])
title(sprintf("Raw: Acc=%.2f", results(modality,1)))
subplot(1,3,2)

imagesc(X_difusion(:,:,i))
xticklabels([])
yticklabels([])
title(sprintf("Diffusion process: Acc=%.2f", results(modality,2)))

subplot(1,3,3)
imagesc(X_reconstructed(:,:,i))
xticklabels([])
yticklabels([])
title(sprintf("Tensor fusion: Acc=%.2f", results(modality,3)))
suptitle(sprintf("Modality %i", modality))
f.Position = [180.0000  343.5000  963.5000  313.5000];
set(gcf,'color','w');
end

% sum of all modalities
Ksum_before = sum(double(X_ten),3);
[confusion,acc] = classify_kernel_AD(Ksum_before,K_y,Z);
[accuracy, specificity, sensitivity] = computeMetrics(confusion);

Ksum_difusion = sum(double(X_difusion),3);
[confusion,acc_difusion] = classify_kernel_AD(Ksum_difusion,K_y,Z);
[accuracy_difusion, specificity_difusion, sensitivity_difusion] = computeMetrics(confusion);

Ksum_tensor= sum(double(X_reconstructed),3);
[confusion,acc_tensor] = classify_kernel_AD(Ksum_tensor,K_y,Z);
[accuracy_tensor, specificity_tensor, sensitivity_tensor] = computeMetrics(confusion);
results(end+1,:) = [accuracy, specificity, sensitivity, accuracy_difusion, specificity_difusion, sensitivity_difusion, accuracy_tensor, specificity_tensor, sensitivity_tensor];
f= figure(i+1);
subplot(1,3,1)
imagesc(Ksum_before)
xticklabels([])
yticklabels([])
title(sprintf("Raw: Acc=%.2f",acc))
subplot(1,3,2)
imagesc(Ksum_difusion)
xticklabels([])
yticklabels([])
title(sprintf("Diffusion process: Acc=%.2f", acc_difusion ))

subplot(1,3,3)
imagesc(Ksum_tensor)
xticklabels([])
yticklabels([])
title(sprintf("Tensor fusion: Acc=%.2f", acc_tensor))
suptitle(sprintf("Sum of all modalities "))
f.Position = [180.0000  343.5000  963.5000  313.5000];
set(gcf,'color','w');

% close all;
results_round =  round(results  * 10000) / 100;
% save(sprintf('results/result_ADMCI_%i.mat', exp), 'results_round');
% writematrix(results_round, (sprintf('results/result_ADMCI_%i.csv', exp)));
% break
end
% function [normalizedData] = minMaxNormalize(data)
%     %MINMAXNORMALIZE Normalizes data using the min-max scaling to [0, 1]
%     %   data: Input data matrix (examples in rows, features in columns)
%     %   normalizedData: Output matrix of normalized data
%     %   minVals: Vector of minimum values for each feature
%     %   maxVals: Vector of maximum values for each feature
%     
%     minVals = min(data); % Compute the minimum value for each feature
%     maxVals = max(data); % Compute the maximum value for each feature
%     
%     % Prevent division by zero if any feature has the same min and max
%     range = maxVals - minVals;
%     range(range == 0) = 1;
%     
%     % Normalize data to [0, 1]
%     normalizedData = (data - minVals) ./ range;
% end