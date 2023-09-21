import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='bagel-tensorflow',
    version='2.2.0',
    author='Zhenyu Zhong',
    author_email='nczzy1997@gmail.com',
    license='MIT',
    description='Implementation of Bagel in TensorFlow 2',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/alumik/bagel-tensorflow',
    packages=setuptools.find_packages(include=['bagel', 'bagel.*']),
    platforms='any',
    install_requires=[
        'pandas',
        'scikit-learn',
        'tensorflow~=2.13.0',
        'tensorflow-probability~=0.21.0',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.10',
)
