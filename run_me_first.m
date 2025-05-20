%% Fed-MVKM Project Initialization Script
% This script initializes the Fed-MVKM project environment by adding all necessary
% folders to the MATLAB path. Run this script before executing any other scripts
% in the project.
%
% Project Structure:
%   - Dataset/     : Contains all datasets and data preprocessing functions
%   - com_func/   : Common utility functions
%   - evaluation/ : Performance evaluation metrics
%   - tools/      : Helper tools and utilities
%   - Functions/  : Core algorithm implementations
%   - measurement/: Performance measurement tools
%
% Author: Kristina P. Sinaga
% Email: kristinasinaga41@gmail.com
% Date: Last updated May 2024
%
% This work was supported by the National Science and Technology Council, 
% Taiwan (Grant Number: NSTC 112-2118-M-033-004)
%
% Usage:
%   1. Set this directory as your current MATLAB directory
%   2. Run this script
%   3. Proceed with running the main algorithms
%
% Note: This script will modify your MATLAB path for the current session.
% The changes are not permanent and will be reset when you restart MATLAB.
%--------------------------------------------------------------------------

% Clear workspace and command window
clear all; close all; clc

% Add the root project directory to path
fprintf('Initializing Fed-MVKM project environment...\n');
addpath(pwd);

% Add required project directories
directories = {...
    'Dataset',    'Dataset folder with all required data files';
    'com_func',   'Common utility functions';
    'evaluation', 'Performance evaluation metrics';
    'tools',      'Helper tools and utilities';
    'Functions',  'Core algorithm implementations';
    'measurement','Performance measurement tools'
};

% Add each directory and its subdirectories to path
for i = 1:size(directories, 1)
    dir_name = directories{i,1};
    dir_desc = directories{i,2};
    
    if exist(dir_name, 'dir')
        cd(dir_name);
        addpath(genpath(pwd));
        cd('..');
        fprintf('Added %s: %s\n', dir_name, dir_desc);
    else
        warning('Directory "%s" not found. Some functionality may be limited.', dir_name);
    end
end

fprintf('\nInitialization complete!\n');
fprintf('You can now run the Fed-MVKM algorithms.\n');
fprintf('For examples, see DHA_Fed_MVKM.m\n');
