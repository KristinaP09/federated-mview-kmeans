%% MVKM-ED: Rectified Gaussian Kernel Multi-View K-Means Clustering Algorithm
% This implementation presents a multi-view clustering approach using rectified
% Gaussian kernels for distance computation. The algorithm effectively handles
% multiple views of data while automatically learning view importance weights.
%
% Authors: Kristina P. Sinaga
% Date: Oct. 20, 2023
% Version: 1.0
% Tested on: MATLAB R2020a
%
% Copyright (c) 2023-2024 Kristina P. Sinaga
% Contact: kristinasinaga41@gmail.com

%
% Algorithm Overview:
%   1. Initialize cluster centers and view weights
%   2. Compute beta parameters for distance adaptation
%   3. Update cluster memberships
%   4. Update cluster centers
%   5. Update view weights
%   6. Iterate until convergence
%
% Input Parameters:
%   X           - (cell array) Multi-view dataset matrices, where X{h} is the h-th view
%   cluster_num - (integer) Number of clusters to form
%   points_view - (integer) Number of data views
%   Aalpah     - (scalar) Exponent parameter to control view weights
%   Bbetah     - (scalar) Distance control parameter
%   dh         - (vector) View-specific dimension parameters
%
% Output Parameters:
%   A          - (cell array) Cluster centers for each view
%   V          - (vector) View weights
%   U          - (matrix) Cluster membership matrix
%   index      - (vector) Cluster assignments
%   Param_beta - (vector) Computed beta parameters
%
% Internal Variables:
%   s            - Number of views (shorthand for points_view)
%   c            - Number of clusters (shorthand for cluster_num)
%   data_n       - Number of data points
%   Param_alpha  - View weight control parameter
%   time         - Current iteration counter
%   max_time     - Maximum number of iterations
%   obj_MVKM_ED  - Objective function values over iterations
%
% References:
%   [1] K. P. Sinaga, "Rectified gaussian kernel multi-view k-means 
%       clustering," arXiv, 2024.
%
%-------------------------------------------------------------------------------------------------------------------

function [A, V, U, index, Param_beta] = MVKM_ED(X, cluster_num, points_view, Aalpah, Bbetah, dh)
% Initialize algorithm parameters
s = points_view;            % Number of views
c = cluster_num;           % Number of clusters
data_n = size(X{1},1);     % Number of data points
Param_alpha = Aalpah;      % View weight control parameter
Param_beta = Bbetah;       % Distance control parameter

%% INITIALIZATION PHASE
fprintf('Starting MVKM-ED algorithm initialization...\n');

% Initialize cluster centers using random selection
initial = randperm(data_n,c);     % Random indices for initial centers
A = cell(1,s);                    % Initialize cluster centers cell array
for h = 1:s
    A{h} = X{h}(initial,:);      % Set initial centers for each view
end

% Initialize view weights with uniform distribution
V = ones(1,s) ./ s;              % Equal weights for all views

% Initialize iteration variables
time = 1;                        % Current iteration
max_time = 100;                  % Maximum iterations allowed
obj_MVKM_ED = zeros(1,max_time); % Store objective values

%% MAIN ITERATION LOOP
fprintf('Starting main iteration loop...\n');
while time <= max_time
    fprintf('--------------  Iteration %d  ----------------------\n', time);
    
    %% Step 1: Compute Beta Parameters
    % Beta controls the influence of distance in each view
    Param_beta = zeros(1,s);
    for h = 1:s
        Param_beta(h) = abs(sum(mean(X{h})*c/(time*data_n)));
    end

    %% Step 2: Update Membership Matrix
    % Calculate membership values for each data point
    U = zeros(data_n,c);         % Initialize membership matrix
    membership_values = zeros(data_n,c);
    
    for k = 1:c
        view_distances = zeros(data_n,s);
        for h = 1:s
            % Calculate distances in feature space
            dist = bsxfun(@minus,X{h},A{h}(k,:)).^2;
            kernel_dist = exp(-Param_beta(h).*sum(dist,2));
            rectified_dist = 1 - kernel_dist;
            % Weight by view importance
            view_distances(:,h) = (V(h)^Param_alpha) .* rectified_dist;
        end
        membership_values(:,k) = sum(view_distances,2);
    end
    
    % Assign each point to nearest cluster
    [~, assignments] = min(membership_values,[],2);
    for i = 1:data_n
        U(i,assignments(i)) = 1;
    end
%--------------------------------------------------------------------------
%% Step 3: Update Cluster Centers
    % Update cluster centers for each view
    new_A = cell(1,s);
    for h = 1:s
        for k = 1:c
            numerator = zeros(1,size(X{h},2));
            denominator = 0;
            
            for i = 1:data_n
                % Calculate kernel distances
                dist = sum((X{h}(i,:) - A{h}(k,:)).^2);
                kernel_val = exp(-Param_beta(h) * dist);
                weighted_kernel = (V(h)^Param_alpha) * kernel_val;
                
                % Update sums for center calculation
                numerator = numerator + weighted_kernel * U(i,k) * X{h}(i,:);
                denominator = denominator + weighted_kernel * U(i,k);
            end
            
            % Compute new center
            new_A{h}(k,:) = numerator / denominator;
        end
    end
    A = new_A;  % Update centers

    %% Step 4: Update View Weights
    % Calculate numerator terms for each view
    V_terms = zeros(1,s);
    for h = 1:s
        view_cost = 0;
        for k = 1:c
            for i = 1:data_n
                if U(i,k) > 0
                    dist = sum((X{h}(i,:) - A{h}(k,:)).^2);
                    kernel_dist = exp(-Param_beta(h) * dist);
                    view_cost = view_cost + U(i,k) * (1 - kernel_dist);
                end
            end
        end
        V_terms(h) = (1/view_cost)^(1/(Param_alpha-1));
    end
    
    % Normalize weights
    V = V_terms / sum(V_terms);

    %% Compute Objective Function
    % Calculate current objective value
    obj_current = 0;
    for h = 1:s
        view_obj = 0;
        for k = 1:c
            for i = 1:data_n
                if U(i,k) > 0
                    dist = sum((X{h}(i,:) - A{h}(k,:)).^2);
                    kernel_dist = exp(-Param_beta(h) * dist);
                    view_obj = view_obj + U(i,k) * (1 - kernel_dist);
                end
            end
        end
        obj_current = obj_current + (V(h)^Param_alpha) * view_obj;
    end
    obj_MVKM_ED(time) = obj_current;

    % Display progress
    fprintf('MVKM-ED: Iteration %d, Objective = %.6f\n', time, obj_MVKM_ED(time));

    % Check convergence
    if time > 1 && abs(obj_MVKM_ED(time) - obj_MVKM_ED(time-1)) <= 1e-4
        % Get final cluster assignments
        [~, index] = max(U,[],2);
        fprintf('------------ Algorithm converged after %d iterations -----------\n\n', time);
        break;
    end
    
    time = time + 1;
end

% If maximum iterations reached without convergence
if time >= max_time
    [~, index] = max(U,[],2);
    fprintf('Warning: Maximum iterations reached without convergence\n');
end

end
