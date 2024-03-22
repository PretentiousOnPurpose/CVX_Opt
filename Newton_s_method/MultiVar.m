clc;
clear all;
close all;

rng(102);

%% Generating Data
A = randn(5, 5);
A = eye(5);
X = randi([1, 5], 5, 1);

Y = A * X;

%% Optimization Algorithm
X0 = randi([1, 5], 5, 1);

Xk = X0;

while (1)
    grad_F_Xk = -2 * A * (Y - A * Xk);
    hess_F_Xk = diag(2 * A * A);

    Xk_1 = Xk - (hess_F_Xk)\grad_F_Xk;
    Xk = Xk_1;
end

%% Results