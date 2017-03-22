from distutils.core import setup, Extension
from setuptools import setup, Extension

config = {
    'include_package_data': True,
    'description': 'FIDDLE: An integrative deep learning framework for functional genomic data inference.',
    'download_url': 'https://github.com/thechurchmanlab/FIDDLE',
    'version': '0.1',
    'packages': ['fiddle'],
    'setup_requires': [],
    'install_requires': [],
    # 'install_requires': ['numpy', 'tqdm', 'scipy'],
    # 'install_requires': ['numpy', 'tqdm', 'scipy', 'bx-python==0.8.0', 'pysam==0.10.0', 'pybedtools==0.7.9'],
    'dependency_links': [],
    'scripts': [],
    'name': 'fiddle'
}

if __name__== '__main__':
    setup(**config)
