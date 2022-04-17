import setuptools
with open("README.md","r") as fp:
    long_description = fp.read()
setuptools.setup(
    name="neurogine",
    version="0.1a",
    author="egeismail",
    author_email="egeismailkosedag@gmail.com",
    description="Simple neural network builder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/egeismail/neurogine",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6.0",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6.0'
)