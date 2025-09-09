from setuptools import setup, find_packages

setup(
    name='pystellarmerger',
    version='0.1',
    description='Python framework for simulating 1D stellar mergers',
    author='Max Heller',
    author_email='max.heller@h-its.org',
    packages=find_packages(where="src"),
    entry_points={
        'console_scripts': [
            'pystellarmerger=PyStellarMerger.pymmams.PyMMAMS:main',  # Pointing to the main function
        ],
    },
    include_package_data=True,
    package_data={
        'PyStellarMerger.data': ['isotopes.pkl', 'units.py'],
    },
    package_dir={"": "src"},
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'numba',
        'mesa_reader',
    ]
)
