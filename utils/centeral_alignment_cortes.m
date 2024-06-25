function K = centeral_alignment_cortes(K)
  N = size(K,1);
  I = ones(N);
  K = K  - 1/N*  I * K - 1/N*  K *I  + 1/(N^2) *  ones(1,N)* K * ones(N,1) * I;
end