# Darksun

A bloodmoon package for handling IROS-related analyses with the WFM coded mask instrument. Darksun provides tools for:
- Run an IROS-based sky fields reconstruction
- Data filtering and output analyses
- Skymaps comparison with chosen catalogues
- Handle output FITS files (saving and/or loading)
- Data plotting

> ⚠️ **Note**: Darksun is under active development. APIs may change between versions.\n
> ⚠️ **Note**: The `bloodmoon` source files are present within the package for development sakes, it will be later removed.


## Installation

### PyPI

```bash
in progress...
```

### From Source

Installing from source is necessary when doing development work. The exact process depends on your platform but will generally require:
- Git
- Python 3.11 or later
- pip
- conda (for environment management)

#### Using Conda

Currently, the dependencies are stored in the `environment.yml` project file. This file is temporary, and during development will be substituted by a `project.toml` file.

```bash
# Clone repository
git clone https://github.com/EdoardoGiancarli/darksun.git
cd darksun

# Create and activate conda environment from environment.yml file
conda env create -f environment.yml
conda activate darksun
```


## Quick Start

```python
...
```

For more, take a look at the `darksun` [demo](demo/demo.ipynb) (still in progress...).


## Development

### Running Tests

Assuming you installed from sources, and your source living into the `darksun` directory, run in your terminal:

```bash
cd darksun
python -m unittest
```

## Contributing

Contributions are welcome!
Before submitting a pull request, ensure all tests are correctly executed.
For bug reports and feature requests, please open an issue.