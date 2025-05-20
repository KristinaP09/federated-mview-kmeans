%% DHA_Fed_MVKM: Demo of Federated Multi-View K-Means Clustering on DHA Dataset
% This script demonstrates the Fed-MVKM algorithm on the DHA (Depth-RGB-Hand Action) dataset,
% which contains multi-view data of hand actions captured using both RGB and depth sensors.
%
% Authors: Kristina P. Sinaga
% Date: Oct. 25, 2023
% Version: 1.0
% Contact: kristinasinaga41@gmail.com
%
% This work was supported by the National Science and Technology Council, 
% Taiwan (Grant Number: NSTC 112-2118-M-033-004)
%
% Required Data Files:
%   - Depth_DHA.mat: Depth sensor data
%   - RGB_DHA.mat: RGB camera data
%   - label_DHA.mat: Ground truth labels
%   - clients_MVDHA.mat: Multi-view data distributed across clients
%   - clients_labelset_DHA.mat: Label sets for each client
%
% Performance Metrics:
%   - NMI (Normalized Mutual Information)
%   - ACC (Accuracy)
%   - F-score
%   - Precision
%   - Recall
%   - AR (Adjusted Rand Index)
%
% Variables:
%   X               - Cell array containing normalized RGB and depth data
%   points_view     - Number of views in the dataset
%   dh             - Dimensions of each view
%   Param.Gamma    - Learning rate for model updates (default: 0.04)
%   Param.Alpha    - View weight control parameters
%   Param.Beta     - Distance control parameters
%   X_sets         - Multi-view datasets distributed across clients
%   label_sets     - Ground truth labels for each client
%   cluster_num    - Number of clusters
%   input_M        - Number of clients
%   n_m            - Number of instances per client
%   c_M            - Number of clusters per client
%
% References:
%   [1] K. P. Sinaga et al., "Federated multi-view k-means clustering," 
%       IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
%
%--------------------------------------------------------------------------

% Clear workspace and suppress warnings
clear all; close all; clc
warning off

%% Data Loading and Preprocessing
% Load multi-view data and labels
load Depth_DHA.mat
load RGB_DHA.mat
load label_DHA.mat

% Normalize both views of the data
X = {normalize_data(RGB_DHA) normlization(Depth_DHA)};

%% Parameter Initialization
% Basic parameters
points_view = length(X);                  % Number of views
dh = [];                                  % Array to store dimensions of each view
for h = 1:points_view                     % Iterate through views
    dh = [dh size(X{h}, 2)];             % Store dimension of each view
end
Param.Gamma = 0.04;                       % Learning rate for model updates

%% Load and Process Client Data
% Load distributed datasets
load clients_MVDHA.mat                    % Multi-view data for each client
load clients_labelset_DHA.mat             % Labels for each client

% Clean up workspace
clear c c_lients cluster_n d_h data_n gamma h i label labelview_set n
clear n nclients partition_num partition_num1 points stat stat_new
clear temp_label temp_new s

% Store client data
X_sets = dataview_set;                    % Client multi-view datasets
label_sets = labelset;                    % Client labels
clear dataview_set labelset

% Process labels
label = label_DHA;
clear RGB_DHA Depth_DHA label_DHA
cluster_num = max(label);                 % Number of clusters

%% Client Parameter Setup
input_M = length(X_sets);                 % Number of clients
c_m = cluster_num*ones(1,input_M);        % Clusters per client

% Calculate instances per client
for m = 1:input_M
    for h = 1:points_view
        n_m(m) = size(X_sets{m}{h}, 1);   % Number of instances for client m
    end
end

% Get ground truth clusters per client
for m = 1:input_M
    c_M(m) = length(unique(label_sets{m})); % Actual clusters in client m
end

%% Algorithm Parameters
Param.Alpha = [15 13 17 18];              % View weight parameters

% Initialize global cluster centers
for h = 1:points_view
    [~, A_global{h}] = litekmeans(X{h}, cluster_num, 'MaxIter', 10, 'Replicates', 100);
end

% Calculate beta parameters for each client
for m = 1:input_M
    for k = 1:cluster_num
        for h = 1:points_view
            % Distance calculations
            Param_beta1 = bsxfun(@minus, X_sets{m}{h}, A_global{h}(k,:));  % Distance matrix
            Param_beta2 = (1/n_m(m)).*(Param_beta1.^2);                    % Normalized squared distance
            Param_beta3{m}{h} = sum(Param_beta2,1);                        % Sum along dimensions
            Param_beta4{m}{h} = max(Param_beta3{m}{h}) - min(Param_beta3{m}{h}); % Range
            Param_beta41{m}{h} = mean(Param_beta3{m}{h});                  % Mean distance
        end
    end
    % Final beta parameter computation
    Param_beta5{m} = mean(Param_beta3{m}{h}).*([Param_beta4{m}{:}]);
    Param.Beta{m} = 10^(2).*(Param_beta5{m});
end

%% Experimental Setup
num_seeds = 8;                            % Number of random seeds
% Random seeds for reproducibility
seeds = [16 17 1118713839 192031222 3139213715 2232082230 3594103811 2633421254 3306027311 1511800150 4024135654];
% Metrics to be computed
metrics_meaning = {'acc'; 'nmi'; 'purity'; 'AR'; 'RI'; 'MI'; 'HI'; 'fscore'; 'precision'; 'recall'};

%% Run Experiments
for l = 1:num_seeds
    rng(seeds(l));                       % Set random seed
    fprintf('The Process is beginning.\n');
    tic
    
    % Run Fed-MVKM algorithm
    [index, A_clients, A_global, V_clients, U_clients, Merged_U, Param_Beta, exper2] = ...
        FedMVKM(X, cluster_num, points_view, X_sets, input_M, c_m, Param.Alpha, Param.Beta, Param.Gamma, dh);
    Param.Beta = Param_Beta;
    toc
    
    % Calculate clustering performance metrics
    [~, nmi, ~] = compute_nmi(label,index);      % Normalized Mutual Information
    ACC = Accuracy(index,double(label));         % Clustering Accuracy
    [f,p,r] = compute_f(label,label);           % F-score, Precision, Recall
    [AR,~,~,~] = RandIndex(label,index);        % Adjusted Rand Index
    
    % Display results
    fprintf('\nl: %d, NMI: %f, ACC: %f, f: %f, p: %f, r: %f, AR: %f', ...
        l, nmi, ACC, f, p, r, AR);
    
    % Store results
    result(l,1) = nmi;  % Normalized Mutual Information
    result(l,2) = ACC;  % Accuracy
    result(l,3) = f;    % F-score
    result(l,4) = p;    % Precision
    result(l,5) = r;    % Recall
    result(l,6) = AR;   % Adjusted Rand Index
    
    disp('-----------------------------end---------------------------');
end

% Calculate final statistics
mean_result = mean(result);              % Mean of all metrics
std_result = std(result);                % Standard deviation of metrics