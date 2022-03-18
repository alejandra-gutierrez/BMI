function [PC, Vs, Ds, V_red, M] = PCA(U, M) 
    % M is the cutoff
    % U is the input matrix [N_trials x N_neurons]

    A = cov(U); % should be 98x 98 -- check
    
    U_mean = mean(U,1);
    [V, D] = eig(A);
    [d, ind] = sort(diag(D));
    Ds = D(ind, ind);
    Vs = V(:, ind);

    V_red = Vs(:, 1:M);
    
    % returns a [N_trials x M] matrix with principal components
    PC = V_red'*(U - U_mean); 
end