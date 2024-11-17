% Step 1: Perform PCA
[coeff, score, latent, ~, explained, ~] = pca(table2array(features));

% Step 2: Compute the contribution of each feature to each principal component
% coeff^2 gives the squared coefficients
% bsxfun(@times, coeff.^2, latent') scales the squared coefficients by the variances (latent)
contribution = bsxfun(@times, coeff.^2, latent');

% Step 3: Sum the contributions across all principal components to get total contribution for each feature
total_contribution = sum(contribution, 2);

% Step 4: Sort the features by total contribution
[sorted_contribution, feature_indices] = sort(total_contribution, 'descend');

% Display the sorted contributions and corresponding feature indices
sorted_contribution
feature_indices
