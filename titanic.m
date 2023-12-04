clear ; close all; clc
warning('off', 'Octave:possible-matlab-short-circuit-operator');

% read the training and testing datasets
XTrain = csvread('data/train_X.csv');
YTrain = csvread('data/train_Y.csv');

XTest = csvread('data/test_X.csv');
YTest = csvread('data/test_Y.csv');

% clean the data: remove the index and the headers
XTrain = XTrain(2:end, :);
XTrain = XTrain(:, 2:end);

YTrain = YTrain(2:end, :);
YTrain = YTrain(:, 2:end);

XTest = XTest(2:end, :);
XTest = XTest(:, 2:end);

YTest = YTest(2:end, :);
YTest = YTest(:, 2:end);

% get the number of observation for each data matrix
m_xtrain = size(XTrain, 1);
m_xtest = size(XTest, 1);

% add aolumn of 1s for vectorized multiplication
XTrain = [ones(m_xtrain, 1) XTrain];
XTest = [ones(m_xtest, 1) XTest];

% get the number of features (columns)
totalFeatures = size(XTrain,2);
theta = zeros(1,totalFeatures);

% get the initial cost
J = calculateCost(theta, XTrain, YTrain);

% meta variables
alpha = 0.003;
iterations = 200000;

# run the gradient descent
[theta, J_history ] = gradientDescent(XTrain, YTrain, theta, alpha,iterations);

# get the prediction agains the test data matrix and get the accumarray
prediction = prediction(theta, XTest);
Accuracy = mean(double(prediction == YTest)) * 100;
fprintf('Accuracy: %f\n', Accuracy);
theta

# plot the cost againts the number of iterations
plot(1:iterations, J_history);
xlabel('Iterations');
ylabel('Cost Function');
title('Cost Function Convergence');

# select a random person to see if he survives or not
num = 65;
test = XTest(num, :)
probability = sigmoid(test * theta');
fprintf('The passanger #%d has the probability of surive of %f\n', num, probability);

