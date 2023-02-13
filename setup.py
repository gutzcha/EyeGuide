"""
This files originate from the "New-Empty-Python-Project-Base" template:
    https://github.com/Neuraxio/New-Empty-Python-Project-Base 
Created by Guillaume Chevalier at Neuraxio:
    https://github.com/Neuraxio 
    https://github.com/guillaume-chevalier 
License: CC0-1.0 (Public Domain)
"""

from setuptools import setup, find_packages

with open('README.md') as _f:
    _README_MD = _f.read()

with open('requirements.txt') as _f:
    _requirements = _f.read()

_VERSION = '0.1'

setup(
    name='EyeGuide',
    version=_VERSION,
    description='Take control using your gaze and facial gestures.',
    long_description=_README_MD,
    classifiers=[
        # TODO: typing.
        "Typing :: Typed"
    ],
    url='https://github.com/gutzcha/EyeGuide',
    download_url='https://github.com/.../.../tarball/{}'.format(_VERSION),  # TODO.
    author='Yizhaq Goussha',
    author_email='gutzcha@gmail.com',
    packages=find_packages(include=['project*']),
    test_suite="testing",
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"],
    install_requires=[_requirements],
    include_package_data=True,
    license='mit',
    keywords='gaze, action classification, facial gestures, eye tracking'
)

