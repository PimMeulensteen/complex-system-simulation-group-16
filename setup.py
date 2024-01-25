from setuptools import setup, find_packages


setup(
    name="css_dla",
    version="1.0",
    packages=find_packages(),
    install_requires = [
        "numpy",
        "matplotlib",
        "powerlaw"
    ],
)