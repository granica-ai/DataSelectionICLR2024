from numpy.polynomial.hermite import hermgauss
import numpy as np


def density(x):
    """
    Standard normal density function in 1D.
    """
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-x ** 2 / 2)


def joint_pdf(x, y):
    """
    Standard normal joint density function in 2D.
    """
    return (1 / (2 * np.pi)) * np.exp(-0.5 * (x ** 2 + y ** 2))


def joint_pdf_3d(x, y, z):
    """
    Standard normal joint density function in 3D.
    """
    return ((2 * np.pi) ** (-3 / 2)) * np.exp(-0.5 * (x ** 2 + y ** 2 + z ** 2))


def trapezoid_integration1D(func, n, bound):
    """
    Calculate the numerical integral of a function using the trapezoid method in 1D.

    Args:
        func: function, the function to integrate.
        n: int, The number of sub-intervals to use.
        bound: float, The bound of the integral.

    Returns:
        float, The numerical integral of the function.
    """
    a = -bound
    b = bound
    if n < 1:
        raise ValueError("Number of subintervals (n) must be >= 1")

    h = (b - a) / n  # Width of each sub-interval
    integral = (func(a) + func(b)) / 2.0  # Initialize with the endpoint values

    for i in range(1, n):
        x_i = a + i * h
        integral += func(x_i) * density(x_i)

    integral *= h  # Multiply by the width of the sub-intervals

    return integral


def expectation_trapez_2d(f, n, trapez_limit):
    """
    Calculate the expectation of a function using the trapezoidal rule in 2D.

    Args:
        f: function, The function for which to calculate the expectation.
        n: int, The number of divisions to use in each dimension.
        trapez_limit: float, The limit of the integral.

    Returns:
        float, The numerical integral of the function.
    """
    # Define integration limits
    x_lower = -trapez_limit
    x_upper = trapez_limit
    y_lower = -trapez_limit
    y_upper = trapez_limit

    # Define the number of divisions for each dimension
    num_divisions_x = n  # Number of divisions along the x-axis
    num_divisions_y = n  # Number of divisions along the y-axis

    # Calculate the step sizes for each dimension
    dx = (x_upper - x_lower) / num_divisions_x
    dy = (y_upper - y_lower) / num_divisions_y

    # Precompute function values and joint PDF on all N^2 points
    function_values = np.zeros((num_divisions_x + 1, num_divisions_y + 1))
    joint_pdf_values = np.zeros((num_divisions_x + 1, num_divisions_y + 1))
    for i in range(num_divisions_x + 1):
        x = x_lower + i * dx
        for j in range(num_divisions_y + 1):
            y = y_lower + j * dy
            function_values[i, j] = f(x, y)
            joint_pdf_values[i, j] = joint_pdf(x, y)

    # Initialize the integral sum
    integral_sum = 0.0

    # Perform the double integration using the trapezoidal rule
    for i in range(num_divisions_x):
        x0 = x_lower + i * dx
        x1 = x_lower + (i + 1) * dx

        for j in range(num_divisions_y):
            y0 = y_lower + j * dy
            y1 = y_lower + (j + 1) * dy

            # Calculate the trapezoidal rule contribution for this cell
            integral_sum += 0.25 * (x1 - x0) * (y1 - y0) * (
                    function_values[i, j] * joint_pdf_values[i, j]
                    + function_values[i + 1, j] * joint_pdf_values[i + 1, j]
                    + function_values[i, j + 1] * joint_pdf_values[i, j + 1]
                    + function_values[i + 1, j + 1] * joint_pdf_values[i + 1, j + 1]
            )

    return integral_sum


def expectation_richardson_trapezoidal_2d(f, n, trapez_limit, m=4):
    """
    Perform Richardson extrapolation to improve the accuracy of the trapezoidal rule in 2D.

    Args:
        f: function, The function for which to calculate the expectation.
        n: int, The number of divisions to use in each dimension.
        trapez_limit: float, The limit of the integral.
        m: int, The number of iterations to perform.
    """
    # Calculate initial trapezoidal estimates
    I = expectation_trapez_2d(f, n, trapez_limit)
    I_half = expectation_trapez_2d(f, n * 2, trapez_limit)

    # Perform Richardson extrapolation
    p = 2  # Order of accuracy for the trapezoidal rule
    I_extrapolated = I_half + (I_half - I) / (2 ** p - 1)

    # Refine the extrapolated result using m iterations
    for _ in range(m):
        n *= 2
        I = I_extrapolated
        I_half = expectation_trapez_2d(f, n * 2, trapez_limit)
        I_extrapolated = I_half + (I_half - I) / (2 ** p - 1)

    return I_extrapolated


def expectation_gauss_hermite_2d(f, n):
    """
    Calculate the expectation of a function using Gauss-Hermite quadrature in 2D.

    Args:
        f: function, The function for which to calculate the expectation.
        n: int, The number of quadrature points to use.
    """
    # Obtain the Gauss-Hermite quadrature points and weights
    x, wx = hermgauss(n)
    y, wy = hermgauss(n)

    # Initialize the sum
    integral_sum = 0.0

    # Perform the Gauss-Hermite quadrature
    for i in range(n):
        for j in range(n):
            # Compute the function value at the quadrature point
            value = f(np.sqrt(2) * x[i], np.sqrt(2) * y[j])
            # Compute the product of weights
            weight_product = wx[i] * wy[j]
            # Accumulate the sum
            integral_sum += value * weight_product

    # Normalize by dividing by (pi^n)
    integral_approximation = integral_sum / (np.pi ** 1)

    return integral_approximation


def expectation_trapez_3d(f, n, trapez_limit):
    """
    Calculate the expectation of a function using the trapezoidal rule in 3D.

    Args:
        f: function, The function for which to calculate the expectation.
        n: int, The number of divisions to use in each dimension.
        trapez_limit: float, The limit of the integral.

    Returns:
        float, The numerical integral of the function.
    """
    # Define integration limits
    x_lower = -trapez_limit
    x_upper = trapez_limit
    y_lower = -trapez_limit
    y_upper = trapez_limit
    z_lower = -trapez_limit
    z_upper = trapez_limit

    # Define the number of divisions for each dimension
    num_divisions_x = n  # Number of divisions along the x-axis
    num_divisions_y = n  # Number of divisions along the y-axis
    num_divisions_z = n  # Number of divisions along the z-axis

    # Calculate the step sizes for each dimension
    dx = (x_upper - x_lower) / num_divisions_x
    dy = (y_upper - y_lower) / num_divisions_y
    dz = (z_upper - z_lower) / num_divisions_z

    # Precompute function values and joint PDF on all N^3 points
    function_values = np.zeros((num_divisions_x + 1, num_divisions_y + 1, num_divisions_z + 1))
    joint_pdf_values = np.zeros((num_divisions_x + 1, num_divisions_y + 1, num_divisions_z + 1))

    for i in range(num_divisions_x + 1):
        x = x_lower + i * dx
        for j in range(num_divisions_y + 1):
            y = y_lower + j * dy
            for k in range(num_divisions_z + 1):
                z = z_lower + k * dz
                function_values[i, j, k] = f(x, y, z)
                joint_pdf_values[i, j, k] = joint_pdf_3d(x, y, z)

    # Initialize the integral sum
    integral_sum = 0.0

    # Perform the triple integration using the trapezoidal rule
    for i in range(num_divisions_x):
        x0 = x_lower + i * dx
        x1 = x_lower + (i + 1) * dx

        for j in range(num_divisions_y):
            y0 = y_lower + j * dy
            y1 = y_lower + (j + 1) * dy

            for k in range(num_divisions_z):
                z0 = z_lower + k * dz
                z1 = z_lower + (k + 1) * dz

                # Calculate the trapezoidal rule contribution for this cell
                integral_sum += 0.125 * (x1 - x0) * (y1 - y0) * (z1 - z0) * (
                        function_values[i, j, k] * joint_pdf_values[i, j, k]
                        + function_values[i + 1, j, k] * joint_pdf_values[i + 1, j, k]
                        + function_values[i, j + 1, k] * joint_pdf_values[i, j + 1, k]
                        + function_values[i + 1, j + 1, k] * joint_pdf_values[i + 1, j + 1, k]
                        + function_values[i, j, k + 1] * joint_pdf_values[i, j, k + 1]
                        + function_values[i + 1, j, k + 1] * joint_pdf_values[i + 1, j, k + 1]
                        + function_values[i, j + 1, k + 1] * joint_pdf_values[i, j + 1, k + 1]
                        + function_values[i + 1, j + 1, k + 1] * joint_pdf_values[i + 1, j + 1, k + 1]
                )

    return integral_sum


def expectation_gauss_hermite_3d(f, n):
    """
    Calculate the expectation of a function using Gauss-Hermite quadrature in 3D.

    Args:
        f: function, The function for which to calculate the expectation.
        n: int, The number of quadrature points to use.

    Returns:
        float, The numerical integral of the function.
    """
    x, wx = hermgauss(n)
    y, wy = hermgauss(n)
    z, wz = hermgauss(n)

    # Initialize the sum
    integral_sum = 0.0

    # Perform the Gauss-Hermite quadrature
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # Compute the function value at the quadrature point
                value = f(np.sqrt(2) * x[i], np.sqrt(2) * y[j], np.sqrt(2) * z[k])
                # Compute the product of weights
                weight_product = wx[i] * wy[j] * wz[k]
                # Accumulate the sum
                integral_sum += value * weight_product

    # Normalize by dividing by (pi^n)
    integral_approximation = integral_sum / (np.pi ** 1.5)

    return integral_approximation


def expectation_2d(f, n, method, trapez_limit):
    """
    Calculate the expectation of a function using the specified method in 2D.

    Args:
        f: function, The function for which to calculate the expectation.
        n: int, The number of quadrature points to use.
        method: str, The method to use for integration.
        trapez_limit: float, The limit of the integral.

    Returns:
        float, The numerical integral of the function.
    """
    assert method in ['GH', "Trapez", 'TrapezExtrap']
    if method == 'GH':
        return expectation_gauss_hermite_2d(f, n)
    elif method == 'Trapez':
        return expectation_trapez_2d(f, n, trapez_limit)
    elif method == "TrapezExtrap":
        return expectation_richardson_trapezoidal_2d(f, n, trapez_limit)
    else:
        raise NotImplementedError


def expectation_3d(f, n, method, trapez_limit):
    """
    Calculate the expectation of a function using the specified method in 3D.

    Args:
        f: function, The function for which to calculate the expectation.
        n: int, The number of quadrature points to use.
        method: str, The method to use for integration.
        trapez_limit: float, The limit of the integral.

    Returns:
        float, The numerical integral of the function.
    """
    assert method in ['GH', "Trapez"]
    if method == 'GH':
        return expectation_gauss_hermite_3d(f, n)
    elif method == 'Trapez':
        return expectation_trapez_3d(f, n, trapez_limit)
    else:
        raise NotImplementedError
