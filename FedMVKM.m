%% Fed-MVKM: Federated Multi-View K-Means Clustering Algorithm Implementation
% This implementation presents a novel federated learning framework for multi-view 
% clustering that enables collaborative learning across distributed clients while 
% preserving data privacy.
%
% Authors: Kristina P. Sinaga
% Date: Oct. 24, 2023
% Version: 1.0
% Tested on: MATLAB R2020a
%
% Copyright (c) 2023-2024 Kristina P. Sinaga
% Contact: kristinasinaga41@gmail.com
%
% This work was supported by the National Science and Technology Council, 
% Taiwan (Grant Number: NSTC 112-2118-M-033-004)
%
% References:
%   [1] K. P. Sinaga et al., "Federated multi-view k-means clustering," 
%       IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
%   [2] K. P. Sinaga et al., "Rectified gaussian kernel multi-view k-means 
%       clustering," arXiv, 2024.
%
% Example Usage:
%   % Load multi-view data
%   load('Depth_DHA.mat');  % First view
%   load('RGB_DHA.mat');    % Second view
%   load('label_DHA.mat');  % Ground truth labels
%
%   % Set parameters
%   points_view = 2;        % Number of views
%   cluster_num = 5;        % Number of clusters
%   Alpha = [15 13 17 18];  % View weight control parameter
%   Beta = compute_beta(X); % Distance control parameter
%   Gamma = 0.04;          % Model update parameter
%
%   % Run Fed-MVKM
%   [index, A_clients, A_global, V_clients, U_clients, Merged_U, Param_Beta] = ...
%       FedMVKM(X, cluster_num, points_view, X_sets, P, c_lients, Alpha, Beta, Gamma, dh);
%
%-------------------------------------------------------------------------------------------------------------------
% Function Arguments:
%   Inputs:
%     X           - (cell array) Multi-view dataset matrices
%     cluster_num - (integer) Number of clusters to form
%     points_view - (integer) Number of data views
%     X_sets      - (cell array) M clients' multi-view data sets
%     P           - (integer) Number of participating clients
%     c_lients    - (vector) Ground truth number of clusters per client
%     Alpha       - (vector) Exponent parameter to control view weights
%     Beta        - (scalar) Distance control parameter in Euclidean space
%     Gamma       - (scalar) Client model update coefficient
%     dh          - (vector) View-specific dimension parameters
%
%   Outputs:
%     index       - (vector) Cluster assignments for each data point
%     A_clients   - (cell array) Client-specific cluster centers
%     A_global    - (cell array) Global cluster centers
%     V_clients   - (cell array) View weights for each client
%     U_clients   - (cell array) Membership matrices for each client
%     Merged_U    - (matrix) Merged membership matrix
%     Param_Beta  - (array) Computed beta parameters
%     exper2      - (struct) Experimental results and metrics
%
% Internal Variables:
%   s            - Shorthand for points_view
%   c            - Shorthand for cluster_num
%   data_n       - Number of data points
%   Param_Alpha  - Copy of Alpha parameter
%   Param_Beta   - Copy of Beta parameter
%   Param_Gamma  - Copy of Gamma parameter
%   time         - Current iteration counter
%   max_time     - Maximum number of iterations
%   obj_Fed_MVKM - Objective function values over iterations
%
% Algorithm Stages:
%   1. Initialization Stage - Setup of central server
%   2. Client Stage        - Local model optimization
%   3. Federation Stage    - Global model aggregation
%   4. Convergence Stage   - Final model evaluation
%
% Notes:
%   - The algorithm assumes clients have same feature dimensionality but
%     different instances
%   - Clusters across clients are assumed to be unique
%   - Uses exponential kernelization for distance computation
%   - Default maximum iterations is 10 (adjustable)
%-------------------------------------------------------------------------------------------------------------------

function [index, A_clients, A_global, V_clients, U_clients, Merged_U, Param_Beta, exper2] = FedMVKM(X, cluster_num, points_view, X_sets, P, c_lients, Alpha, Beta, Gamma, dh)
% Variable initialization with proper documentation
% Basic parameters
s = points_view;            % Number of views (shorthand)
c = cluster_num;           % Number of clusters (shorthand)
data_n = size(X{1},1);     % Total number of data points

% Parameter copies for internal use
Param_Alpha = Alpha;        % View weight control parameter
Param_Beta = Beta;         % Distance control parameter
Param_Gamma = Gamma;       % Model update parameter

%--------------------------------------------------------------------------
%% INITIALIZATION STAGE: Seamless Central Server
% Initialize global model cluster centers and distribute to clients
%--------------------------------------------------------------------------
disp('- The seamless central server initialize the global cluster centers A...');
for h = 1:s
    % Initialize cluster centers using lite k-means with multiple restarts
    [~, A_global{h}] = litekmeans(X{h},c,'MaxIter', 20,'Replicates',200);
end
disp('- The seamless central server send the initial global model of A to all clients M...');

%--------------------------------------------------------------------------
% Initialize client-specific parameters
P = length(X_sets);                % Number of participating clients
c_lients = c*ones(1,P);           % Number of clusters per client
n_clients = zeros(1,P);           % Array to store number of instances per client

% Calculate number of instances per client
for p = 1:P
    for h = 1:s
        n_clients(p) = size(X_sets{p}{h}, 1);
    end
end

%--------------------------------------------------------------------------
%% INITIALIZATION STAGE 1: CLIENTS
% Initialize client-side parameters and variables
%--------------------------------------------------------------------------
% Initialize view weights for each client
disp('- The mth clients initialize their weighted view V...');
for p = 1:P
    for h = 1:s
        V_clients{p} = ones(1,s)./s;  % Equal weights initially
    end
end

% Initialize iteration variables
time = 1;                             % Current iteration
max_time = 10;                        % Maximum iterations
obj_Fed_MVKM = zeros(1,max_time);     % Store objective values

%--------------------------------------------------------------------------
% Start the iteration
%--------------------------------------------------------------------------
for time = 1: max_time
    fprintf('--------------  The %d Iteration Starts ---------------\n', time); 
    
    %% --------------------------------------------------------------------- %%
    %                           STAGE 1: CLIENTS                               %
    %% --------------------------------------------------------------------- %%  
    % Step 1: Compute the coefficient parameter beta
    % This step calculates the adaptive distance parameter for each client
    for p = 1:P
        for k = 1:c
            for h = 1:s
                % Calculate distance between data points and cluster centers
                Param_beta1 = bsxfun(@minus, X_sets{p}{h}, A_global{h}(k,:));    % Distance matrix
                Param_beta2 = (1/n_clients(p)).*(Param_beta1.^2);                % Normalized squared distance
                Param_beta3{p}{h} = sum(Param_beta2,1);                         % Sum along dimensions
                Param_beta4{p}{h} = max(Param_beta3{p}{h}) - min(Param_beta3{p}{h}); % Range of distances
            end
        end
        % Compute final beta parameter for each client
        Param_beta5{p} = mean(Param_beta3{p}{h}).*([Param_beta4{p}{:}]);
        Param_Beta{p} = 10^(1)*(Param_beta5{p});
    end

    %--------------------------------------------------------------------------
    % Step 2: Compute the memberships U
    % This step updates the cluster membership for each data point
    for p = 1:P
        U8_clients{p} = [];     % Initialize membership matrix
        
        for k = 1:c
            for h = 1:s
                % Calculate distance-based membership values
                U1_clients{p} = bsxfun(@minus, X_sets{p}{h}, A_global{h}(k,:));   % Distance to cluster center
                U2_clients{p} = U1_clients{p}.^2;                                 % Squared distance
                U3_clients{p} = -Param_Beta{p}(h).*U2_clients{p};                % Scaled distance
                U4_clients{p} = (1-exp(U3_clients{p}));                          % Exponential kernel
                U5_clients{p}{h} = ((V_clients{p}(h)).^Param_Alpha(p)).*(sum(U4_clients{p},2)); % View-weighted membership
            end
            U6_clients{p} = [U5_clients{p}{:}];           % Concatenate view memberships
            U7_clients{p} = sum(U6_clients{p},2);         % Sum memberships
            U8_clients{p} = [U8_clients{p} sum(U7_clients{p},2)]; % Store final membership
        end
    end
 
 
 for p = 1:P
     U_clients{p} = zeros(n_clients(p),c);
     for i = 1:n_clients(p)
         [val{p}, idx{p}] = min(U8_clients{p}(i,:));
         U_clients{p}(i, idx{p})=1;
     end
 end  
  
%--------------------------------------------------------------------------
% Step 3: Compute the cluster centers A
 for p = 1:P
     
     for h = 1:s
         
         for k = 1:c_lients(p)
             dist_clients1{p} = bsxfun(@minus, X_sets{p}{h}, A_global{h}(k,:));
             dist_clients2{p} = dist_clients1{p}.^2;
             exper1{p}{h} = dist_clients2{p};
             dist_clients3{p} = -Param_Beta{p}(h).*dist_clients2{p};
             exper2{p}{h}=exp(dist_clients3{p});
             dist_clients{p}{h}  = sum((-exp(dist_clients3{p})),2);
             exper3{p}{h} = bsxfun(@minus, 1, dist_clients{p}{h});
             
         end
         
     end
 end
 
   for p = 1:P
      
      for h = 1:s
          
          for k = 1:c_lients(p)
              A_clients{p}{h}(k,:) = (V_clients{p}(h).^(Param_Alpha(p))...
                  .*X_sets{p}{h}'* (U_clients{p}(:,k)).*...
                  (-dist_clients{p}{h}'))/(V_clients{p}(h).^(Param_Alpha(p))...
                  .*sum(U_clients{p}(:,k)).*(-dist_clients{p}{h}')) ;
              
          end
      end
      
  end    
 
%--------------------------------------------------------------------------
%% Step 4: Update the weighted view V
 for p = 1:P
      
      for h = 1:s

          for k = 1:c
              V1_clients{p}{h} = bsxfun(@minus, X_sets{p}{h}, A_global{h}(k,:));
              V2_clients{p}{h}(:,k) = sum(V1_clients{p}{h}.^2,2);
              V3_clients{p}{h} = -Param_Beta{p}(h).*V2_clients{p}{h};
              V4_clients{p}{h} = exp(V3_clients{p}{h});
              V5_clients{p}{h} = bsxfun(@minus, 1, V4_clients{p}{h});
              V6_clients{p}{h} = bsxfun(@times, U_clients{p}(:,k),V5_clients{p}{h}(:,k)); 
              V7_clients{p}{h} = sum((Param_Alpha(p).*V6_clients{p}{h}),1);
              
              
          end
          V8_clients{p}=[V7_clients{p}{:}];
          V9_clients{p} = (1./V8_clients{p}).^(1/(Param_Alpha(p)-1));
          V10_clients{p} = sum(V9_clients{p},2);

          
      end
      New_V_clients{p} = bsxfun(@rdivide,V9_clients{p}, V10_clients{p});

  end   
  
%--------------------------------------------------------------------------

%% --------------------------------------------------------------------- %%
    %        STAGE 2: Federated Learning or Seamless Central Server            %
%% --------------------------------------------------------------------- %% 
    fprintf('- The FL server received the A model of m clients at %d iteration \n', time); 
    
    % Aggregate the A model of all clients M
    % This step combines the local models into a global model
    Sum_A_m1 = A_clients{1};                  % First client's model
    Sum_A_m2 = A_clients{2};                  % Second client's model
    
    disp('- The FL server starting to aggregate the A model of m clients...');
    for h = 1:s
        % Weighted average of client models based on their data sizes
        New_A_Global1{h} = (Sum_A_m1{h}./n_clients(1) + Sum_A_m2{h}./n_clients(2));
        % Update global model using the learning rate Gamma
        New_A_Global{h} = A_global{h} - (Param_Gamma.*New_A_Global1{h});
    end

    %--------------------------------------------------------------------------
    %% --------------------------------------------------------------------- %%
    %                     STAGE 3: Evaluation                                   %
    %--------------------------------------------------------------------------
    % Computing the objective value 
    disp('- Computing the objective values...');
    for p = 1:P
        for h = 1:s
            for k = 1:c
                % Calculate exponential distances for objective function
                D_exponent1 = bsxfun(@minus,X_sets{p}{h},A_clients{p}{h}(k,:)).^2;
                D_exponent2 = exp(-Param_Beta{p}(h)*sum(D_exponent1,2)); 
                D_exponent3{p}{h}(:,k) = bsxfun(@minus,1,D_exponent2);
            end
        end
    end    

    % Calculate objective function value for each client
    for p = 1:P
        obj{p} = zeros(s,1);
        for h = 1:s
            obj{p}(h) = obj{p}(h) + (V_clients{p}(h).^Param_Alpha(p) .* ...
                sum(sum(U_clients{p}.*D_exponent3{p}{h})));
        end
    end

    % Aggregate objective values across all clients
    for p = 1:P
        obj_Fed_MVKM_h{p} = sum(obj{p});
    end
    obj_Fed_MVKM_h_p = [obj_Fed_MVKM_h{:}];
    obj_Fed_MVKM(time) = sum(obj_Fed_MVKM_h_p);

    %--------------------------------------------------------------------------
    % Check convergence criteria 
    disp('- Check the convergences criteria...');
    fprintf('Fed-MVKM: Iteration count = %d, Fed-MVKM = %f\n', time, obj_Fed_MVKM(time));
    % if time > 1 && (abs(obj_Fed_MVKM(time) - obj_Fed_MVKM(time-1)) <= 1e-4)
    if time == max_time  
       for p = 1:P
     New_U8_clients{p} = [];
     
     for k = 1:c
         
         for h = 1:s
             
             New_U1_clients{p} = bsxfun(@minus, X_sets{p}{h}, A_global{h}(k,:));
             New_U2_clients{p} = New_U1_clients{p}.^2;
             New_U3_clients{p} = -Param_Beta{p}(h).*New_U2_clients{p};
             New_U4_clients{p} = (1-exp(New_U3_clients{p}));
             New_U5_clients{p}{h} = ((V_clients{p}(h)).^Param_Alpha(p)).*(sum(New_U4_clients{p},2));

         end
         New_U6_clients{p} = [New_U5_clients{p}{:}];
         New_U7_clients{p} = sum(New_U6_clients{p},2);
         New_U8_clients{p} = [New_U8_clients{p} sum(New_U7_clients{p},2)];          
     end
     
     
 end
 
 
 for p = 1:P
     New_U_clients{p} = zeros(n_clients(p),c);
     for i = 1:n_clients(p)
         [val{p}, idx{p}] = min(New_U8_clients{p}(i,:));
         New_U_clients{p}(i, idx{p})=1;
     end
 end 
 
    
    % Concatenated the memberships of all clients M
    Merged_U = [New_U_clients{1};New_U_clients{2}];
    
    index = [];
    for i = 1:data_n
        [numed idxed] = max(Merged_U(i,:));
        index = [index; idxed];
    end
    disp('------------ The Iteration has finished.-----------');
   break;
end
time = time +1;
    
end
end


