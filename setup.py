from setuptools import setup, find_packages

setup(
    name='SGanalysis',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for spatial analysis of specific data.',
    packages=find_packages(),
    install_requires=[
        'geopandas',
        'rasterio',
        'shapely',
        'tifffile',
        'numpy'
    ],
)
