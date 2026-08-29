from setuptools import setup, find_packages

setup(
    name="Recallspection",
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
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
)
