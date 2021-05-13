from distutils.core import setup

setup(name='experiment',
    version='0.1',
    description='Tools for running experiments.',
    author='Howard Huang',
    packages=['experiment'],
    install_requires=['dill','tqdm','matplotlib','scipy','scikit-optimize'],
    extras_require={
        'dev': ['pytest','pytest-cov','pdoc3']
    }
)
