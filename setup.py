"""
EntroPit - Probabilistic Dungeon Generation
Setup configuration for package installation
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="entropit",
    version="0.1.0",
    author="EntroPit Team",
    author_email="your.email@example.com",
    description="Probabilistic dungeon generation powered by thermodynamic computing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/entropit",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/entropit/issues",
        "Documentation": "https://github.com/yourusername/entropit/blob/main/docs/",
        "Source Code": "https://github.com/yourusername/entropit",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Games/Entertainment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "cuda11": ["jax[cuda11]"],
        "cuda12": ["jax[cuda12]"],
    },
    entry_points={
        "console_scripts": [
            "entropit-quickstart=examples.quickstart:main",
            "entropit-ui=examples.interactive_ui:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

