# feature-specific-contribution
Different methods to compute the specific contribution of a groups of features X = [A, B, C] to predict a target Y.

**Objective**

We aim at computing the specific contribution of different groups of features contained in X (ex: X = [A, B, C], with A, B, C having different dimensionality) 
to predict each dimension of a target Y.

(Note that would like to keep track of the contribution for each dimension of Y). 

We use the three methods below. 
All three methods are coded in the `src/get_contrib.py` file, and are called by the `run_experiments()` function contained in `src/run_experiments.py`.
The three functions take as inputs: `X`, `y`, `hiearchy` (e.g [0, 0, 0, 1, 1, 2], with 0 (resp. 1 and 2) refering to the indices of A (resp. B and C) in X),
the predictive scikit-learn `model`, and the three parameters `scale` (whether to scale per group), `n_repeats` (for permutation importance) 
and `pca` (number of components if we apply PCA to each group before fitting).

**Method 1: "permute_current"**

Function `contrib_permute_current()` in `get_contrib.py`
+ (optional) Scale each group of features separately
+ (optional) PCA on each group
+ orthogonalise
+ concatenate features
+ fit on train
+ compute permutation importance, for each feature group separately, on test
+ re-iterate on folds


**Method 2: "permute_below"**

Function `contrib_permute_below()` in `get_contrib.py`

+ (optional) Scale each group of features separately
+ (optional) PCA on each group
+ concatenate features
+ fit on train
+ compute permutation importance, for (`+` refers to concatenation)
    * the first group A: `importance[A] = r - r_with_A_shuffled`
    * the first + second group: `importance(A+B) = r - r_with_A_and_B_shuffled`
    * the first + second + third group: `importance(A+B+C) = r - r_with_A_and_B_and_C_shuffled`
    * ... 
    * all groups shuffled
+ extract specific contribution for each level
    * `contrib_A = importance(A)`
    * `contrib_B = importance(A+B) - importance of first group (A)`
    * ... 
+ re-iterate on folds


**Method 3: "concat"**

Function `contrib_concat()` in `get_contrib.py`

+ (optional) Scale each group of features separately
+ (optional) PCA on each group
+ concatenate features
+ in order to predict Y, fit in hiearchical order and compute scores of: (`+` refers to concatenation)
    * the first group A: `r(A)`
    * the first + second group: `r(A+B)`
    * ... 
+ extract specific contribution for each level
    * `contrib_A = r(A)`
    * `contrib_B = r(A+B) - r(A)`
    * `contrib_C = r(A+B+C) - r(A+B)`
    
    
    
**Parameters**

Each function computes the contribution of feature groups (e.g [A, B, C]),
indicated in the parameter `hierarchy`.
Ex: if hierarchy = [0, 0, 0, 1, 1, 2] (n_groups=3), and y is shape (n, 10)
(y_dim=10), each method would return an array of shape (n_groups, y_dim) = (3, 10), giving
the contribution of each group to the prediction of a particular dimension.

Each function takes as input:

* **X** : array of shape `(n, x_dim)`
  - features
* **y** : array of shape `(n, y_dim)`
  - target
* **hierarchy** : array of int of shape (x_dim). 
    - Indices of the group for each feature.
    - Ex: [0, 0, 0, 1, 1, 2] refers to 3 groups, the first with dim=3, second with dim=2, last dim=1
* **model** : sklearn model, optional, by default RidgeCV(np.logspace(-2, 8, 20)). 
    - Predictive model used
      - to predict y given X,
      - if exp="permute_one", the model is also used to orthogonlize features
* **scale** : bool, optional. 
    - Whether to scale each group of features (independantly) before fitting, by default True
* **n_repeats** : int, optional. 
    - Number of repeats for exp "permute_xxx", by default 50.
    - Scores are always averaged across repeats.
* **pca** : int, optional.
    - If > 0, PCA is applied to each group of features 
    (independantly) before fitting, refers to the number 
    of components to use in the PCA, by default 20.

**Returns**

Each function outputs:

* **importance** : an array of shape `(n_groups, y_dim)`, 
    with n_groups = len(unique(hierarchy))
    `importance[i, j]` gives the contribution of features
    group i for th prediction of y_j.
