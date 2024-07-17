import numpy as np
from ConvertSupport import convert_support

def setup_data():
    from_grid = np.arange(0, np.pi / 2, 0.1)
    to_grid = from_grid + 0.001
    to_grid[-1] = to_grid[-1] - 0.002
    mu = np.sin(from_grid)
    phi = np.vstack((np.sin(from_grid), np.cos(from_grid))).T
    phi1 = phi[:, 0].reshape(-1, 1)
    Cov = np.dot(phi, phi.T)
    return from_grid, to_grid, mu, phi, phi1, Cov

def test_convert_support_mu():
    from_grid, to_grid, mu, _, _, _ = setup_data()
    result_mu = convert_support(from_grid, to_grid, mu=mu)
    np.testing.assert_allclose(mu, result_mu, atol=2e-3)

def test_convert_support_phi():
    from_grid, to_grid, _, phi, _, _ = setup_data()
    result_phi = convert_support(from_grid, to_grid, phi=phi)
    np.testing.assert_allclose(phi, result_phi, atol=2e-3)

def test_convert_support_cov():
    from_grid, to_grid, _, _, _, Cov = setup_data()
    result_cov = convert_support(from_grid, to_grid, Cov=Cov)
    np.testing.assert_allclose(Cov, result_cov, atol=1e-3)

if __name__ == "__main__":
    pytest.main()
