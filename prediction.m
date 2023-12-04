function prediction = prediction(theta, X)
    prediction = sigmoid(X * theta') >=0.5;
end
