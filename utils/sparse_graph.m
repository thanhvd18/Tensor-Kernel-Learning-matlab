function S = sparse_graph(W_i, k)
N = size(W_i,1);
S = zeros(size(W_i));
indies = zeros(size(W_i));
for row=1:N
    [row_sorted,indies(row,:)] =sort(W_i(row,:),'descend');
%     for i=1:N
        for j=indies(row,2:2+k)
            S(row,j) = W_i(row,j)/sum(row_sorted(2:2+k));
        end
%     end
end

S = S + S' - diag(diag(S));

end