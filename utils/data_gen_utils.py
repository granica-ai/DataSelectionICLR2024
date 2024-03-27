import numpy as np
from utils.misc_utils import sigmoid


def generate_theta(p, seed, scale=1):
    """
    Generate a random theta.

    Args:
        p: int, number of features.
        seed: int, random seed.
        scale: float, norm of the generated theta.

    Returns:
        theta: np.array, random theta of size (1, p).
    """
    np.random.seed(seed)
    theta = np.random.randn(p)
    theta = theta - theta.mean()
    theta = scale * theta / np.linalg.norm(theta)

    return theta[None, :]


def generate_X(n, p, seed):
    """
    Generate random data with isotropic covariance.

    Args:
        n: int, number of samples.
        p: int, number of features.
        seed: int, random seed.

    Returns:
        X: np.array, random data of size (n, p).
    """
    np.random.seed(seed)
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=np.eye(p, p), size=n)
    return X


def generate_labels(X, theta, seed, label_method, label_method_a, label_method_b):
    """
    Generate binary labels. Supports well-specified and specific mis-specified models.

    Args:
        X: np.array, features.
        theta: np.array, true parameter.
        seed: int, random seed.
        label_method: str, method to generate labels.
        label_method_a: float, parameter for label_method.
        label_method_b: float, parameter for label_method.

    Returns:
        y: np.array, binary labels.
    """
    # Calculate logits.
    z = np.dot(X, theta.squeeze())

    # Calculate probabilities.
    if label_method == 'logistic':
        prob = sigmoid(z)
    elif label_method == 'misspec':
        prob = np.where(z >= 0, label_method_a, label_method_b)
    elif label_method == 'misspec2':
        conditions = [z < -0.5, (z >= -0.5) & (z < 0), (z >= 0) & (z < 0.5), z >= 0.5]
        values = [0.3, label_method_b, label_method_a, 0.7]
        prob = np.select(conditions, values)
    elif label_method == 'misspec3':
        conditions = [np.abs(z) < 0.5, z >= 0.5, z <= -0.5]
        values = [0.5, 0.95, 0.05]
        prob = np.select(conditions, values)
    else:
        raise NotImplementedError

    np.random.seed(seed)
    y = np.random.binomial(1, prob.flatten())

    return y


def generate_data(dataN, dataP, data_theta_norm, data_seed, label_method, label_method_a, label_method_b):
    """
    Generate synthetic data. Supports well-specified and specific mis-specified models.

    Args:
        dataN: int, number of samples.
        dataP: int, number of features.
        data_theta_norm: float, norm of the true parameter.
        data_seed: int, random seed.
        label_method: str, method to generate labels.
        label_method_a: float, parameter for label_method.
        label_method_b: float, parameter for label_method.

    Returns:
        X_total: np.array, features of size (dataN, dataP).
        y_total: np.array, binary labels of size (dataN, 1).
        theta: np.array, true parameter of size (1, dataP).
    """
    # generate isotropic covariates
    X_total = generate_X(dataN, dataP, data_seed)
    # generate random true parameter theta
    theta = generate_theta(dataP, data_seed + 1, data_theta_norm)
    # generate labels based on <X, theta>
    y_total = generate_labels(X_total, theta, data_seed + 2, label_method, label_method_a, label_method_b)

    return X_total, y_total, theta


def accuracy_score(y_true, y_pred):
    """
    Compute accuracy score.

    Args:
        y_true: np.array, true labels.
        y_pred: np.array, predicted labels.

    Returns:
        accuracy: float, accuracy score.
    """
    return np.mean(y_true == y_pred)


def generate_similar(theta, corr, seed):
    """
    Given a float corr and a vector theta of size (1, N), generates theta_rot, s.t:
    <theta, theta_rot> = corr
    ||theta|| = ||theta_rot||

    Args:
        theta: np.array, vector of size (1, N).
        corr: float, correlation between theta and theta_rot.
        seed: int, random seed.

    Returns:
        theta_rot: np.array, rotated theta of size (1, N).
    """

    if corr == 1:
        return theta
    assert 0 <= corr < 1, "Correlation should be between 0 and 1"
    assert theta.shape == (1, len(theta)), "Theta should be a row vector"

    angle = np.arccos(corr)
    norm = np.sqrt(theta @ theta.T).squeeze()
    theta_norm = theta / norm
    np.random.seed(seed)
    random = np.random.normal(size=theta.shape)
    random_orthogonal = random - (random @ theta_norm.T) * theta_norm
    random_orthogonal_norm = random_orthogonal / np.linalg.norm(random_orthogonal)
    _ = random_orthogonal_norm @ theta_norm.T

    theta_rot = random_orthogonal_norm + (1 / np.tan(angle)) * theta_norm
    theta_rot = theta_rot / np.linalg.norm(theta_rot)
    theta_rot = norm * theta_rot

    return theta_rot
