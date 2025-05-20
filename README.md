# Fed-MVKM: Federated Multi-View K-Means Clustering

## Overview
Fed-MVKM is a novel federated learning framework for multi-view clustering that enables collaborative learning across distributed clients while preserving data privacy. This algorithm effectively handles heterogeneous data distributions across clients and achieves robust clustering performance through a privacy-preserving mechanism.

## Key Features
- Privacy-preserving federated multi-view clustering
- Adaptive view weight learning mechanism
- Seamless central server coordination
- Support for heterogeneous data distributions
- Client-side local model optimization
- Global model aggregation strategy

## Requirements
- MATLAB R2020a or later
- Statistics and Machine Learning Toolbox
- Parallel Computing Toolbox (recommended)

## Installation
```bash
git clone https://github.com/yourusername/Fed-MVKM.git
cd Fed-MVKM
```

## Usage
The main function can be called as follows:
```matlab
[index, A_clients, A_global, V_clients, U_clients, Merged_U, Param_Beta, exper2] = ...
    FedMVKM(X, cluster_num, points_view, X_sets, P, c_lients, Alpha, Beta, Gamma, dh)
```

### Parameters
- `X`: Multi-view dataset (sample-view space)
- `cluster_num`: Number of clusters
- `points_view`: Number of data views
- `X_sets`: M clients' multi-view data sets
- `Alpha`: Exponent parameter to control weights of V
- `Beta`: Coefficient parameter for distance control
- `Gamma`: Coefficient parameter for clients' model updating
- `dh`: View-specific dimension parameters

## Algorithm Stages
1. **Initialization Stage**: Seamless central server setup
2. **Client Stage**: Local model optimization
   - Coefficient parameter computation
   - Membership calculation
   - Cluster center updates
   - View weight updates
3. **Federation Stage**: Global model aggregation
4. **Convergence Stage**: Final model evaluation

## Example
```matlab
% Load multi-view data
load('Depth_DHA.mat');
load('RGB_DHA.mat');
load('label_DHA.mat');

% Set parameters
points_view = 2;
cluster_num = 5;
Alpha = [15 13 17 18];
Beta = compute_beta(X);
Gamma = 0.04;

% Run Fed-MVKM
[index, ~, ~, ~, ~, ~, ~, ~] = FedMVKM(X, cluster_num, points_view, X_sets, ...
    P, c_lients, Alpha, Beta, Gamma, dh);
```

## Citation
If you use this code in your research, please cite our paper:
```bibtex
@article{sinaga2024federated,
  title={Federated Multi-View K-Means Clustering},
  author={Sinaga, Kristina P. and others},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
- Kristina P. Sinaga
- Email: kristinasinaga41@gmail.com

## References
1. [Federated multi-view k-means clustering](https://ieeexplore.ieee.org/abstract/document/10810504) - IEEE TPAMI 2024
2. Rectified gaussian kernel multi-view k-means clustering - arXiv 2024

## Acknowledgments
- This work was supported by the National Science and Technology Council, Taiwan (Grant Number: NSTC 112-2118-M-033-004)
- Special thanks to collaborators and contributors

## Note
The code has been tested on MATLAB R2020a. Performance on other versions may vary.
