function [beta, K_combine,bc_test] = MKL_wrap(K,Y,Z)
C = 1;
verbose=1;

options.algo='svmclass'; % Choice of algorithm in mklsvm can be either
                         % 'svmclass' or 'svmreg'
%------------------------------------------------------
% choosing the stopping criterion
%------------------------------------------------------
options.stopvariation=0; % use variation of weights for stopping criterion 
options.stopKKT=0;       % set to 1 if you use KKTcondition for stopping criterion    
options.stopdualitygap=1; % set to 1 for using duality gap for stopping criterion

%------------------------------------------------------
% choosing the stopping criterion value
%------------------------------------------------------
options.seuildiffsigma=1e-2;        % stopping criterion for weight variation 
options.seuildiffconstraint=0.1;    % stopping criterion for KKT
options.seuildualitygap=0.01;       % stopping criterion for duality gap

%------------------------------------------------------
% Setting some numerical parameters 
%------------------------------------------------------
options.goldensearch_deltmax=1e-1; % initial precision of golden section search
options.numericalprecision=1e-8;   % numerical precision weights below this value
                                   % are set to zero 
options.lambdareg = 1e-8;          % ridge added to kernel matrix 

%------------------------------------------------------
% some algorithms paramaters
%------------------------------------------------------
options.firstbasevariable='first'; % tie breaking method for choosing the base 
                                   % variable in the reduced gradient method 
options.nbitermax=500;             % maximal number of iteration  
options.seuil=0;                   % forcing to zero weights lower than this 
options.seuilitermax=10;           % value, for iterations lower than this one 

options.miniter=0;                 % minimal number of iterations 
options.verbosesvm=0;              % verbosity of inner svm algorithm 

%
% Note: set 1 would raise the `strrep`
%       error in vectorize.dll
%       and this error is not able to fix
%       because of the missing .h libraay files
% Modify: MaxisKao @ Sep. 4 2014
options.efficientkernel=0;         % use efficient storage of kernels 


%%
N = size(K,2);
M = size(K,1);
% K = zeros(N,N,M);
% for i=1:size(K_X,1)
%     K(:,:,i) = K_X(i,:,:);
%     K_temp = K(:,:,i);
%     K_temp(1:N+1:end) = ones(N,1);
%     K(:,:,i) = K_temp;
% end
% Y = squeeze(K_y(1,1,:));
% Y = double(categorical(Y))*2-3;

train_index = [Z{1} Z{2}] +1; %train + val, rescale 0-(N-1) -> 1-N
test_index = Z{3} +1;
train_index = double(train_index);
test_index = double(test_index);

K_train = K(train_index,train_index,:);
y_train = Y(train_index);

K_test = K(test_index,train_index,:);
y_test = Y(test_index);



tic
[beta,w,b,posw,story,obj] = mklsvm(K_train,y_train,C,options,verbose);
timelasso=toc

K_train_combine = 0;
for i=1: length(beta)
    if i==1
        K_train_combine = beta(i)*K_train(:,posw,i);
    else
        K_train_combine = K_train_combine + beta(i)*K_train(:,posw,i);
    end 
end


ypred=K_train_combine*w+b;
% 
bc_train=mean(sign(ypred)==y_train)

%%

 K_test_combine = 0;
 for i=1: length(beta)
     if i==1
            K_test_combine = beta(i)*K_test(:,:,i);
        else
            K_test_combine = K_test_combine + beta(i)*K_test(:,:,i);
        end
 end
 
K_combine = 0;
 for i=1: length(beta)
     if i==1
            K_combine = beta(i)*K(:,:,i);
        else
            K_combine = K_combine + beta(i)*K(:,:,i);
        end
 end
 
 
 K_test_combine = K_test_combine(:,posw,:);
 ypred=K_test_combine*w+b;
 % 
 bc_test=mean(sign(ypred)==y_test)

%%
% [C, acc] = classify_kernel_AD(K_combine,K_y,Z)