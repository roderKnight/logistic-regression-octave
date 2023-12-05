function prediction = prediction(theta, X, B)
    prediction = sigmoid(X * theta' + B) >= 0.5;
end
