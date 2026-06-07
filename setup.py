from setuptools import setup, find_packages

setup(
    name="opwa",
    version="0.1.0",
    description="Orthogonal Plug-in Weather Adapter - A1 Minimum Viable Version",
    author="OPWA Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "diffusers>=0.24.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
    ],
)
