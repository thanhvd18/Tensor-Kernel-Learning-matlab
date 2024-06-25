
function  K = centeral_alignment_cris(K)
  n = size(K,1);
  H = eye(n) - ones(n, n) / n;
  K = H *  K * H;
end
