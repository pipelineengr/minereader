from setuptools import setup, find_packages

setup(
    name="minereader",
    version="0.1.0",
    description="Small project to learn about ML through Python in ore grade estimation",
    packages=find_packages(),
    
    # find_packages() automatically finds all directories containing __init__.py
    # __init__.py files are currently empty but the imports can be moved there for
    # proper package maintenance
    
    python_requires="~=3.12",
    install_requires=[
        "torch",
        "torch_geometric",
        "fastapi",
        "uvicorn",
        "pydantic",
        "pandas",
        "numpy",
        "scikit-learn",
        "plotly",
        "scipy",
        "joblib",
        "pykrige",
    ],
    entry_points={
        "console_scripts": [
            "minereader=minereader.cli:main",
            # format: "command_name=module:function"
            # This tells pip: when someone types 'minereader',
            # run the main() function in cli.py
        ],
    },
)