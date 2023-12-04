function [J] = calculateCost(theta, input, output)
    m = length(output);
    J = 0;
    # Calculate z For Sigmoid
    z = input * theta';
    J = (-output' * log(sigmoid(z)) - (1-output') * log(1-
    sigmoid(z)))/m;
end
