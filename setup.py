"""Install Dplython."""


from setuptools import setup, find_packages


setup(
    name="dplython",
    version="0.0.5",
    description="Dplyr-style operations on top of pandas DataFrame.",
    url="https://github.com/dodger487/dplython",
    download_url="https://github.com/dodger487/dplython/tarball/0.0.5",
    packages=find_packages(),
    license="MIT",
    keywords="pandas data dplyr",
    package_data={"dplython": ["data/diamonds.csv"]},
    package_dir={"dplython": "dplython"},
    install_requires=["numpy", "pandas", "six"],
    author="Chris Riederer",
    author_email="OfficialChrisEmail@gmail.com",
    maintainer="Chris Riederer",
    maintainer_email="OfficialChrisEmail@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Environment :: Web Environment",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering",
    ]
)