function [theta,J_history] = gradientDescent(input, output, theta, alpha, iterations)
    m = length(output);
    J_history = zeros(iterations, 1);
    for i = 1:iterations
         z = input * theta';
         theta = theta - (sigmoid(z) - output)' * input * (alpha/m);
         cost = calculateCost(theta,input, output);
         J_history(i) = cost;

         if mod(i, iterations/10) == 0
            fprintf('Cost after %d iterations is: %f\n', i, cost);
        end

     end
 end
