function err = reconstruction_err(A,B)
err = (norm(A-B)/ norm(A)*100);
end
