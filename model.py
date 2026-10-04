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

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    # TODO: Standardize features then add the bias column to form the design matrix.
    X = standardize_features(X, mean, std)
    X = add_bias_column(X)
    return X

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    # TODO: Return the predicted target vector from X and weights
    y_hat = X @ weights
    return y_hat

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    # TODO: Return the average of squared residuals as a scalar float.
    n = len(y_true)
    se = 0
    for i in range(n):
        se += (y_true[i] - y_pred[i]) ** 2
    mse = se / n
    return mse

# Step 9 - mse_gradient
import numpy as np
def mse_gradient(X, y_true, y_pred):
    # TODO: Return the analytic MSE gradient w.r.t. weights: (2/n) X^T (y_pred - y_true)
    X = np.asarray(X)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    w_grad = 2/n * (X.T @ (y_pred-y_true))
    return w_grad

# Step 10 - normal_equation
import numpy as np
def normal_equation(X, y):
    # TODO: Solve for the closed-form least-squares weights via the normal equation.
    X = np.asarray(X)
    y = np.asarray(y)
    w = np.linalg.inv(X.T @ X) @ X.T @ y 
    return w

# Step 11 - initialize_weights
import numpy as np
def initialize_weights(n_features, seed=None):
    # TODO: Return (n_features,) weights sampled from N(0, 0.01)
    if seed is not None:
        np.random.seed(seed = seed)
    w = np.random.normal(loc = 0.0, scale = 0.01, size = n_features)
    return w

# Step 12 - gd_step
import numpy as np
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    # TODO: return the updated weight vector after one MSE gradient step
    X = np.asarray(X)
    y = np.asarray(y)
    d_w = mse_gradient(X, y, X @ weights)
    weights = weights - lr * d_w
    return weights

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    # TODO: return the pair (train_loss, val_loss) as MSE floats
    train_loss = mse_loss(y_train, X_train @ weights)
    val_loss = mse_loss(y_val, X_val @ weights)
    return (train_loss, val_loss)

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    # TODO: Update best weights and patience counter; signal stop when val loss stalls...
    stop = False
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        best_weights = weights.copy()
    else:
        wait += 1
        stop = (wait >= patience)
    return best_val_loss, wait, best_weights, stop

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    # TODO: Build the initial training-state dictionary for the GD epoch loop.
    w = initialize_weights(n_features, seed = seed)
    state = {
        "weights": w.copy(),
        "best_weights": w.copy(),
        "best_val_loss": np.inf,
        "wait": 0,
        "train_losses": [],
        "val_losses": [],
        "stopped":False
    }
    return state

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state.

    Args:
        state: Dict with keys weights, best_weights, best_val_loss, wait,
            stopped, train_losses, val_losses.
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        lr: Learning rate (float).
        patience: Early-stopping patience (int).

    Returns:
        Updated state dict.
    """
    # TODO: Take one GD step, log train/val losses, refresh early-stopping fields...
    weights = gd_step(X_train, y_train, state["weights"], lr)
    state["weights"] = weights
    train_loss, val_loss = epoch_train_val_losses(X_train, y_train, X_val, y_val, weights)
    state["train_losses"].append(train_loss)
    state["val_losses"].append(val_loss)
    best_val_loss, wait, best_weights, stop = update_early_stop_state(val_loss,
                                                state["best_val_loss"],state["wait"] ,weights, state["best_weights"],patience)
    state["best_weights"] = best_weights
    state["best_val_loss"] = best_val_loss
    state["wait"] = wait 
    state["stopped"] = stop 
    return state

# Step 17 - train_batch_gd
def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    # TODO: Train weights with full-batch GD for up to epochs, with early stopping.
    n_features = X_train.shape[1]
    weights = initialize_weights(n_features, seed = seed)
    state = init_training_state(n_features, seed = seed)
    for epoch in range(epochs):
        weights = gd_step(X_train, y_train, weights, lr)
        train_loss, val_loss = epoch_train_val_losses(X_train, y_train, X_val, y_val, weights)
        state = run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience)
        if state["stopped"] == True:
            break 
    return state["weights"], state["train_losses"], state["val_losses"]

# Step 18 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: Compute the mean absolute error between true targets and predictions
    n = len(y_true)
    mae = 0
    for i in range(n):
        if y_true[i] < y_pred[i]:
            mae -= y_true[i] - y_pred[i]
        else:
            mae += y_true[i] - y_pred[i]
    return mae * 1/n

# Step 19 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    # TODO: Return the root mean squared error between y_true and y_pred.
    n = len(y_true)
    MSE = 0
    for i in range(n):
        MSE += (y_true[i] - y_pred[i]) ** 2
    RMSE = (1/n * MSE) ** 0.5
    return RMSE

# Step 20 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute the coefficient of determination R^2.
    n = len(y_true)
    res, tot = 0.0, 0.0
    sum = 0.0
    for i in range(n):
        sum += y_true[i]
    y_mean = sum / n
    for i in range(n):
        res += (y_true[i] - y_pred[i]) ** 2
        tot += (y_true[i] - y_mean) ** 2
    if tot == 0: return float('nan')
    R_square = 1 - res / tot 
    return R_square

# Step 21 - evaluate_regression
def evaluate_regression(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, and R^2 into one metrics dictionary for test-set reporting.
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r_squared(y_true, y_pred)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }

# Step 22 - learning_curve_data
import numpy as np
def learning_curve_data(train_losses, val_losses):
    # TODO: Return epoch indices and loss series for external plotting...
    n = len(train_losses)
    train_losses = np.asarray(train_losses)
    val_losses = np.asarray(val_losses)
    epochs = []
    for i in range(1, n+1):
        epochs.append(i)
    return epochs, train_losses.tolist(), val_losses.tolist()

# Step 23 - weights_l2_distance
import numpy as np
def weights_l2_distance(w_gd, w_closed):
    # TODO: Compute the L2 distance between two weight vectors
    w_gd = np.asarray(w_gd)
    w_closed = np.asarray(w_closed)
    L2 = np.sqrt(np.sum((w_gd - w_closed) ** 2))
    return L2

# Step 24 - create_lr_model
def create_lr_model(learning_rate=0.01, epochs=1000, patience=50, seed=0):
    # TODO: Build the initial LinearRegressionGD-style model dictionary...
    model = {}
    model["learning_rate"] = learning_rate
    model["epochs"] = epochs
    model["patience"] = patience
    model["seed"] = seed
    model["weights"] = None
    model["normal_weights"] = None 
    model["mean"] = None 
    model["std"] = None 
    model["train_losses"] = []
    model["val_losses"] = []
    return model

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

