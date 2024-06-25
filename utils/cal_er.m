function fit = cal_er(X,P)
% extract from cp_als function
normX = norm(X);
normresidual = sqrt( normX^2 + norm(P)^2 - 2 * innerprod(X,P) );
fit = 1 - (normresidual / normX); %fraction explained by model
end