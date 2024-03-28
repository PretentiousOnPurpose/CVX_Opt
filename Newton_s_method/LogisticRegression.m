clc;
clear all;
close all;

rng(100);

%% Data
M = 1000;

Np = 2;

Y = randi([0, 1], M, 1);
Z = Y .* 10;
X = Z + 2 * randn(M, 1);

%% Logistic Regression Model
W = randn(Np, 1);

X1 = [X, ones(M, 1)];

Y_hat = Sigmoid(X1 * W);

%% Newton's Method for Optimization

numEpochs = 1000;

for iter_e = 1: numEpochs
    testModel(W, X1, Y);

    delF = zeros(M, Np);
    delF_2 = zeros(Np, Np);
    
    Y_hat = Sigmoid(X1 * W);
    
    loss_factor = (1 - Y) .* Y_hat - Y .* (1 - Y_hat);
    sigGradFn = Y_hat .* (1 - Y_hat);

    for iter_np = 1: Np
        delF(:, iter_np) = X1(:, iter_np) .* loss_factor;
    end
    
    delF = mean(delF);

    for iter_M = 1: M
        delF_2_tmp = (X1(iter_M, :).' * X1(iter_M, :)) .* Y_hat(iter_M);

        delF_2 = delF_2 + delF_2_tmp;
    end

    delF_2 = delF_2 ./ M;

    W = W - delF_2 \ delF.';
end

function sigOut = Sigmoid(sigIn)
    sigOut = 1 ./ (1 + exp(-sigIn));
end

function testModel(W, X, Y)
    Y_hat = Sigmoid(X * W);
    
    Y_hat(Y_hat > 0.5) = 1;
    Y_hat(Y_hat < 1) = 0;
    
    disp(['Accuracy: ', num2str(sum(Y_hat == Y) / numel(Y))]);
end