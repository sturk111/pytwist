# pytwist

pytwist is a package intended for use by experimental physicists working on the fabrication and characterization of twisted graphene based devices.  The package includes three model classes: twisted bilayer (TBG), twisted double bilayer (TDBG), and mirror symmetric twisted trilayer (TTG). Each class is capable of accepting experimental parameters like twist angle, strain, and displacement field as input and computing the resulting band structures and densities of states in just a few lines of code.

The calculations implement the continuum model formalism introduced by Bistritzer and MacDonald (2011).  See the following publications for more details:

Bistritzer and MacDonald PNAS 108 (30) 12233-12237.

S. Turkel et al. Science 376 (6589), 193-199.

C. Rubio-Verdu*, S. Turkel* et al. Nat. Phys. 18(2), 196-202.

## Usage

```python
#create TDBG model
tdbg = TDBG(1.05,0,0,0)

#plot and return band structure
evals_m, evals_p, kpath = tdbg.solve_along_path()

#plot and return density of states
pdos = tdbg.solve_PDOS()

#return local density of states
m = tdbg.solve_LDOS()
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
