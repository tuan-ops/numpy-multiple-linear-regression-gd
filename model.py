"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    """ Bước này dùng để xáo trộn dữ liệu trước khi tạo train/test/validation
    tuy nhiên phải giữ đúng cặp dữ liệu đầu vào X và nhãn y
        Tức là phải tạo 1 hoán vị ra 1 indices mà cả X và y đều đổi vì nếu bộ 
        dữ liệu quy định X -> y chứ không thể có y khác ở đây
        Ý tưởng:
            - Tạo ra 1 indices để có thể đều hoán trộn X và y bằng hàm 
            np.random.RandomState(seed = seed) rồi dùng hàm np.random.permutation
            để tạo ra hoán vị bất kì -> indices
            - Sau khi đã có 1 list các index đã bị xáo trộn rồi thì sẽ gán vào cả X và y
    """
    rng = np.random.RandomState(seed = seed)
    indices = rng.permutation(X.shape[0])
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    return X_shuffled, y_shuffled

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    """
        Bước này để chia dữ liệu thành các tệp train/ validation/ test
        sau khi dữ liệu đã trộn 
        Đầu bài cho tỉ lệ giữa tập train và tập val với dữ liệu ban đầu
        Yêu cầu trả về 3 tập riêng biệt (X, y)
    """
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    X_train = X[:n_train, :]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val , :]
    y_val = y[n_train:n_train + n_val ]
    X_test = X[n_train + n_val:, :]
    y_test = y[n_train + n_val:]
    return (X_train, y_train, X_val, y_val, X_test, y_test)

# Step 3 - compute_feature_stats
import numpy as np
def compute_feature_stats(X):
    """
        Bước này sẽ chuẩn hóa cho toàn bộ tập train tính mean và std 
        cho từng cột (từng feature) sau đó sẽ áp dụng công thức chuẩn hóa
    """
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis =0)
    std = np.where(std == 0.0 , 1.0, std)
    z = (X - mean) / std
    return mean, std

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    # TODO: Apply z-score normalization using precomputed training mean and std.
    X = (X - mean) / std
    return X

# Step 5 - add_bias_column
def add_bias_column(X):
    """
    Thêm bias/intercept vào X nếu không thêm thì sẽ luôn ép mô hình
    đi qua gốc tọa độ

    """
    n = X.shape[0]
    tmp = np.ones((n, 1))
    X = np.hstack([tmp, X])
    return X

# Step 6 - prepare_design_matrix (not yet solved)
# TODO: implement

# Step 7 - predict_linear (not yet solved)
# TODO: implement

# Step 8 - mse_loss (not yet solved)
# TODO: implement

# Step 9 - mse_gradient (not yet solved)
# TODO: implement

# Step 10 - normal_equation (not yet solved)
# TODO: implement

# Step 11 - initialize_weights (not yet solved)
# TODO: implement

# Step 12 - gd_step (not yet solved)
# TODO: implement

# Step 13 - epoch_train_val_losses (not yet solved)
# TODO: implement

# Step 14 - update_early_stop_state (not yet solved)
# TODO: implement

# Step 15 - init_training_state (not yet solved)
# TODO: implement

# Step 16 - run_one_epoch (not yet solved)
# TODO: implement

# Step 17 - train_batch_gd (not yet solved)
# TODO: implement

# Step 18 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 19 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 20 - r_squared (not yet solved)
# TODO: implement

# Step 21 - evaluate_regression (not yet solved)
# TODO: implement

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

