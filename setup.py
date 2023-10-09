from setuptools import find_packages, setup

requirements = [
    "torch",
    "datasets",
    "transformers",
    "numpy",
    "wandb",
    "huggingface-hub>=0.16.4,<0.18",
]

setup(
    name="transformercompression",
    version="0.0.1",
    author="James Hensman, Max Croci, Saleh Ashkboos, Marcelo Gennari do Nascimento",
    description="Implementation of methods for compressing transformers",
    packages=find_packages("src"),  # include all packages under src
    package_dir={"": "src"},  # tell distutils packages are under src
    install_requires=requirements,
)
