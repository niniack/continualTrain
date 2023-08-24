from setuptools import setup, find_packages

setup(
    name="continualTrain",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["PyYAML"],
    entry_points={
        "console_scripts": [
            "continual_train=api.train:main",
        ],
    },
)
