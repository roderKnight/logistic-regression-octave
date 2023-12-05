function [W, B, J_history] = gradientDescent(input, output, theta, alpha, iterations)

    m = length(output);
    J_history = zeros(iterations, 1);

    W = theta;
    B = 0;

    for i = 1:iterations
         z = input * W' + B;
         a = sigmoid(z);

         % calculate the cost
         cost = calculateCost(W, input, output);

         % gradient descent
         % derivative of cost function with respect to theta (w)
         dW = (1/m) * (a - output)' * input;
         % derivative of cost function with respect to b
         dB = (1/m) * sum(a - output);

         % update the weigths
         W = W - alpha * dW;
         % update the bias
         B = B - alpha * dB;

         J_history(i) = cost;

         if mod(i, iterations/10) == 0
            fprintf('Cost after %d iterations is: %f\n', i, cost);
        end

     end
 end
