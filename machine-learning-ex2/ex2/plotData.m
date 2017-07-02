function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% Build index of axis values where y==1 and y==0
admit = find(y==1);
reject = find(y==0);

% Plot the training data to show admits and rejects

plot (X(admit,1),X(admit,2), 'k+', 'LineWidth',2,'MarkerSize',7);
plot (X(reject,1),X(reject,2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize',7);


% =========================================================================



hold off;

end
