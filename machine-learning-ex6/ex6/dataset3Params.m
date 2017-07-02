function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Use cross-validation set Xval,yval to determine best C and sigma
% Note: For C and sigma use [0.01 0.03 0.1 0.3 1 3 10 30]

testC = [0.01 0.03 0.1 0.3 1 3 10 30];
testSigma = [0.01 0.03 0.1 0.3 1 3 10 30];

results = [];

for p=1:length(testC)
	for q=1:length(testSigma)
		
		% Trains 64 models over the two for loops
		model = svmTrain(X,y,testC(p),@(x1,x2) gaussianKernel(x1,x2,testSigma(q)));
		predictions = svmPredict(model,Xval);
		testError = mean(double(predictions ~= yval));

		% Store C and sigma parameters and error values in running array
		results = [results; testC(p) testSigma(q) testError];
	end
end

[minError, minIndex] = min(results(:,3));

C = results(minIndex,1);
sigma = results(minIndex,2);




% =========================================================================

end
