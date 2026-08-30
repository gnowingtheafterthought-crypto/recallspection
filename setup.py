from setuptools import setup, find_packages

setup(
    name="recallspection",
    version="18.0.0",
    description="Dual‑core exact memory for AI agents (ExactMemory + SWSTM)",
    author="Sciencedelic Metatech",
    author_email="eliamraell@yandex.com",
    url="https://github.com/sciencedelicmetatech/recallspection",
    packages=find_packages(include=["recallspection", "recallspection.*"]),
    include_package_data=True,
    install_requires=[
        "fastapi",
        "uvicorn",
        "sentence-transformers",
        "scikit-learn",
        "torch",
        "numpy",
        "pydantic",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "ruff",
        ],
    },
    python_requires=">=3.9",  # Changed from 3.8 to match CI matrix
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",  # FIXED
        "Operating System :: OS Independent",
    ],
)