clc;
clear all;
close all;

%% Generating Data
N = 10;
M = 6;

A = randi([-10, 10], M, 1);

X = rand(N, M) .* 20 - 10;
W = randn(N, 1) * 0.1;

Y = X * A + W;

%% Optimization Algorithm
A0 = randi([-10, 10], M, 1);

Ak = A0;
cnt = 0;

while (1)
    grad_F_Ak = 0;
    hess_F_Ak = 0;
    for iter_n = 1: N
        grad_F_Ak_tmp = 2 * (Y(iter_n) - X(iter_n, :) * Ak) .* (-X(iter_n, :));
        hess_F_Ak_tmp = (2 * X(iter_n, :).' .* X(iter_n, :));
        
        hess_F_Ak = hess_F_Ak + hess_F_Ak_tmp;
        grad_F_Ak = grad_F_Ak + grad_F_Ak_tmp;
    end

    hess_F_Ak = hess_F_Ak ./ N;
    grad_F_Ak = grad_F_Ak ./ N;

    Ak_1 = Ak - hess_F_Ak \ grad_F_Ak.';
    
    % if (abs(Ak_1 - Ak) < 0.05)
    %     break;
    % end

    cnt = cnt + 1;
    
    Ak = Ak_1;
end

%% Results