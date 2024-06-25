
%%
function K_norm_ten = cross_diffusion_process(K_ten,kNN,iter)
    m = size(K_ten,3);
    for i=1:m
     K_norm_ten(:,:,i) =  similarity_normalization(squeeze(K_ten(:,:,i)));  
    end
%     K_norm_ten = K_ten;
    
    for i=1:m
     S_ten(:,:,i) =  sparse_graph(squeeze(K_norm_ten(:,:,i)),kNN);  
    end
    
    for t=1:iter
        for i=1:m 
            S_i = squeeze(S_ten(:,:,i));
            K_norm_ten(:,:,i) =   S_i * (1/(m-1) * sum(K_norm_ten(:,:,setdiff(1:m, i)),3))* S_i';
        end
    end
end

function W_i = similarity_normalization(W)
    W_i = 1/2 *eye(size(W,1));
    N = size(W,1);
    
    for row=1:N
        for col=row+1:N
            W_i(row,col) = W(row,col)/(2*sum(W(row, setdiff(1:N, row))));
        end
    end
    
    W_i = W_i + W_i' - diag(diag(W_i));
end






