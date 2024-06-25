
function W_i = similarity_normalization(W)
    W_i = 1/2 *eye(size(W,1));
    N = size(W,1);
    for row=1:N
        for col=1:N
            W_i(row,col) = W(row,col)/(2*sum(W(row, setdiff(1:N, row))));
        end
    end
    
    W_i = W_i + W_i' - diag(diag(W_i));
end

