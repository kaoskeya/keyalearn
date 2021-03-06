import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keyalearn-kaoskeya", # Replace with your own username
    version="0.2.1",
    author="Karthikeyan Sekizhar",
    author_email="k@kaos.agency",
    description="A package to assist in machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kaoskeya/keyalearn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'sklearn'
    ],
    python_requires='>=3.6',
)
