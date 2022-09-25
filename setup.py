import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='bagel-tensorflow',
    version='2.0.2',
    author='AlumiK',
    author_email='nczzy1997@gmail.com',
    license='MIT',
    description='Implementation of Bagel in TensorFlow 2',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AlumiK/bagel-tensorflow',
    packages=setuptools.find_packages(include=['bagel', 'bagel.*']),
    platforms='any',
    install_requires=[
        'pandas',
        'scikit-learn',
        'tensorflow',
        'tensorflow-probability',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='~=3.8',
)
