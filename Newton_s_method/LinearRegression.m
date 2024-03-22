clc;
clear all;
close all;

%% Generating Data
A = 2.951;

X = randi([1, 10], 10, 1);
W = randn(10, 1) * 0.1;

Y = A * X + W;

%% Optimization Algorithm
A0 = randi([-5, 5], 1, 1);

Ak = A0;
cnt = 0;

while (1)
    grad_F_Ak = 2 * (Y - Ak .* X) .* (-X);
    hess_F_Ak = (2 * X .* X);

    Ak_1 = Ak - mean(grad_F_Ak ./ hess_F_Ak);
    
    if (abs(Ak_1 - Ak) < 0.05)
        break;
    end

    cnt = cnt + 1;
    
    Ak = Ak_1;
end

%% Results