"""
Setup script for DEAP Solver Family

Author: LUMEN Team [TEAM-ID: 93912]
"""

from setuptools import setup, find_packages

setup(
    name="deap_family",
    version="1.0.0",
    description="DEAP Solver Family for Timetable Scheduling Optimization - Stage 6.3",
    author="LUMEN Team",
    author_email="lumen@example.com",
    packages=find_packages(),
    install_requires=[
        "deap>=1.3.3",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "sympy>=1.9",
        "networkx>=2.6",
        "pyarrow>=5.0.0",
        "pyyaml>=5.4.0",
    ],
    entry_points={
        "console_scripts": [
            "deap-solver=deap_family.__main__:main",
        ],
    },
    python_requires=">=3.9",
    zip_safe=False,
)
