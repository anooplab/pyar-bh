from setuptools import setup

setup(
    name='pyar-bh',
    version='0.1',
    packages=['pyar_bh'],
    scripts=['scripts/pyar-bh'],
    url='https://github.com/anooplab/pyar-bh',
    license='GPL',
    author='anoop',
    author_email='anoop@chem.iitkgp.ac.in',
    description='Global optimization of molecules using Basinhopping '
                'in SciPy. Just a blackbox implementation.'
)
