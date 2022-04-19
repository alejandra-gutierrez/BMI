function yhat = nonLinModelFun(beta, x)
    N = 4;
    L = size(x, 1);
    b = zeros(N, L);
    b(1, :) = beta(1: L);
    b(2, :) = beta(L+1 : 2*L);
    b(3, :) = beta(2*L+1 : 3*L);
    b(4, :) = beta(3*L+1 : 4*L);


    yhat = b(1, :)*x + b(2, 1);
    % use 1st element of 1st row for cst element of regression (instead of
    % whole column)
end