from setuptools import setup

setup(
    name='pyar-de',
    version='0.1',
    packages=['pyar_de'],
    scripts=['scripts/pyar-de'],
    url='https://github.com/anooplab/pyar-de',
    license='GPL',
    author='anoop',
    author_email='anoop@chem.iitkgp.ac.in',
    description='Global optimization of molecules using Differential Evolution '
                'in SciPy. Just a blackbox implementation.'
)
