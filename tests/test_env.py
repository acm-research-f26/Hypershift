def test_imports():
    import numpy, scipy, sklearn, torch  # noqa: F401
    import hypershift
    assert hypershift.__version__ == "0.1.0"
