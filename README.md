# pytwist
pytwist module
pytwist.Et(theta, theta_s, e, delta)
A matrix that simultaneously rotates and strains the vector on which it operates.

See Bi et al. Phys. Rev. B 100, 035448 (2019).

Parameters:
theta (float) – Rotation angle in radians.

theta_s (float) – Strain angle in radians.

e (float) – Strain magnitude.

delta (float) – Poisson ratio.

Returns:
result

Return type:
2x2 numpy array

pytwist.R(x)
class pytwist.TBGModel(theta, phi, epsilon, a=0.246, beta=3.14, delta=0.16, vf=1, u=0.11, up=0.11, cut=4)
Bases: object

This is the model class for twisted bilayer graphene.

See Bistritzer and MacDonald PNAS 108 (30) 12233-12237.

thetafloat
Twist angle between the top and bottom bilayer in degrees.

phifloat
Heterostrain angle in degrees relative to the atomic lattice.

epsilonfloat
Heterostrain magnitude. For example, a value of 0.01 corresponds to a 1% difference in lattice constants between the top and bottom layers.

afloat
Lattice constant of graphene in nanometers.

betafloat
Two center hopping modulus. See Phys. Rev. B 100, 035448 (2019).

deltafloat
Poisson ratio for graphene.

vffloat
Fermi velocity renormalization constant. Typical values are 1.2 - 1.3.

u, upfloat
Interlayer hopping amplitudes in electron-Volts for AA and AB sites respectively. u != up captures the effect of finite out of plane corrugation of the moire lattice.

cutfloat
Momentum space cut off in units of the moire lattice vector. Larger values will result in a larger Hamiltonian matrix. Convergence is typically attained by cut = 4.

Examples

Construct a model object representing a twisted bilayer graphene device with a twist angle of 1.11 degrees, a strain angle of 15 degrees, a strain magnitude of 0.5%.

tbg = TBGModel(1.11, 15, 0.005)
gen_ham(kx, ky, xi=1)
Generate hamiltonian for a given k-point.

Parameters:
kx (float) – x and y coordinates of the momentum point for which the Hamiltonian is to be generated in inverse nanometers. Note that kx points along Gamma-Gamma of the moire Brillouin zone.

ky (float) – x and y coordinates of the momentum point for which the Hamiltonian is to be generated in inverse nanometers. Note that kx points along Gamma-Gamma of the moire Brillouin zone.

xi (+/- 1) – Valley index.

Returns:
ham

Return type:
numpy matrix, shape (2*Nq, 2*Nq), dtype = complex

Examples

Generate Hamiltonian and solve for eigenstates at the K point of the moire Brillouin zone (kx=ky=0). Eigenvalues and eigenvectors are stored in vals and vecs respectively.

tbg = TTGModel(1.11, 15, 0.005)
ham =  tbg.gen_ham(0,0)
vals, vecs = eigh(ham)
solve_DOS(nk=16, energies=array([-0.1, -0.099, -0.098, -0.097, -0.096, -0.095, -0.094, -0.093, -0.092, -0.091, -0.09, -0.089, -0.088, -0.087, -0.086, -0.085, -0.084, -0.083, -0.082, -0.081, -0.08, -0.079, -0.078, -0.077, -0.076, -0.075, -0.074, -0.073, -0.072, -0.071, -0.07, -0.069, -0.068, -0.067, -0.066, -0.065, -0.064, -0.063, -0.062, -0.061, -0.06, -0.059, -0.058, -0.057, -0.056, -0.055, -0.054, -0.053, -0.052, -0.051, -0.05, -0.049, -0.048, -0.047, -0.046, -0.045, -0.044, -0.043, -0.042, -0.041, -0.04, -0.039, -0.038, -0.037, -0.036, -0.035, -0.034, -0.033, -0.032, -0.031, -0.03, -0.029, -0.028, -0.027, -0.026, -0.025, -0.024, -0.023, -0.022, -0.021, -0.02, -0.019, -0.018, -0.017, -0.016, -0.015, -0.014, -0.013, -0.012, -0.011, -0.01, -0.009, -0.008, -0.007, -0.006, -0.005, -0.004, -0.003, -0.002, -0.001, 0., 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.05, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069, 0.07, 0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079, 0.08, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089, 0.09, 0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099, 0.1]), xi=1, plot_it=True)
Solves the Hamiltonian for a grid of k points within the first Brillouin zone and computes the full density of states (i.e. momentum integrated number of solutions per unit energy).

Parameters:
nk (int) – The total number of points in the moire Brillouin zone is nk**2. Larger nk corresponds to more accurate and more computationally expensive calculations.

energies (numpy array of floats) – A list of energies in electron-Volts at which to compute the density of states.

xi (+/-1) – Valley index.

plot_it (boolean) – If true a function call will display a plot of the density of states projected onto each layer.

Returns:
dos – Array containing the density of states at each input energy.

Return type:
numpy array, shape (len(energies),)

Examples

Generate and plot the density of states on the top layer for energies at 2 meV increments from -100 to +100 meV.

tbg = TBGModel(1.11, 15, 0.005)
x = np.round(np.arange(-0.1,0.102,0.002),3)
dos = tbg.solve_DOS(energies = x)
plt.plot(x,dos)
solve_along_path(res=16, plot_it=True, return_eigenvectors=False)
Compute the band structure along a path within the moire Brillouin zone. The path is predefined to be K -> Gamma -> M -> K’.

Parameters:
res (float) – The resolution of the cut, defined as the number of points to sample along the line Gamma -> K

plot_it (boolean) – If true a function call will display a plot of the resulting band structure.

return_eigenvectors (boolean) – If true return eigenvectors along the k path.

Returns:
evals_m (numpy array, shape (len(kpath), 2*Nq)) – Eigenvalues along the k path for valley xi = -1. Axis 0 indexes points along the k path. Axis 1 indexes bands.

evals_p (numpy array, shape (len(kpath), 2*Nq)) – Eigenvalues along the k path for valley xi = +1. Axis 0 indexes points along the k path. Axis 1 indexes bands.

evecs_m (numpy array, shape (len(kpath), 2*Nq, 2*Nq)) – Eigenvectors along the k path for valley xi = -1. Axis 0 indexes points along the k path. Axis 1 indexes into the eigenvector. Axis 2 indexes bands. E.g. evecs_m[i,:,j] is the eigenvector for k point i and band j.

evecs_p (numpy array, shape (len(kpath), 2*Nq, 2*Nq)) – Eigenvectors along the k path for valley xi = +1. Axis 0 indexes points along the k path. Axis 1 indexes into the eigenvector. Axis 2 indexes bands. E.g. evecs_p[i,:,j] is the eigenvector for k point i and band j.

kpath (list of shape (2,) numpy arrays) – List of k points along the path. Each point is represented as a two component array in the list.

Examples

Solve for the bands along the path K -> Gamma -> M -> K’, returning eigenvectors.

tbg = TDBGModel(1.11, 15, 0.005)
evals_m, evals_p, evecs_m, evecs_p, kpath = tbg.solve_along_path(return_eigenvectors = True)
The same thing without returning eigenvectors.

evals_m, evals_p, kpath = tbg.solve_along_path()
A higher resolution computation.

evals_m, evals_p, kpath = tbg.solve_along_path(res = 64)
class pytwist.TDBGModel(theta, phi, epsilon, D, a=0.246, beta=3.14, delta=0.16, vf=1, u=0.0797, up=0.0975, cut=4)
Bases: object

This is the model class for twisted double bilayer graphene.

See C. Rubio-Verdu*, S. Turkel* et al. Nat. Phys. 18(2), 196-202 and Koshino Phys. Rev. B 99, 235406 (2019).

thetafloat
Twist angle between the top and bottom bilayer in degrees.

phifloat
Heterostrain angle in degrees relative to the atomic lattice.

epsilonfloat
Heterostrain magnitude. For example, a value of 0.01 corresponds to a 1% difference in lattice constants between the top and bottom layers.

Dfloat
Potential difference between adjacent layers in Volts. This captures the effect of an external displacement field.

afloat
Lattice constant of graphene in nanometers.

betafloat
Two center hopping modulus. See Phys. Rev. B 100, 035448 (2019).

deltafloat
Poisson ratio for graphene.

vffloat
Fermi velocity renormalization constant. Typical values are 1.2 - 1.3.

u, upfloat
Interlayer hopping amplitudes in electron-Volts for AA and AB sites respectively. u != up captures the effect of finite out of plane corrugation of the moire lattice.

cutfloat
Momentum space cut off in units of the moire lattice vector. Larger values will result in a larger Hamiltonian matrix. Convergence is typically attained by cut = 4.

Examples

Construct a model object representing a twisted double bilayer graphene device with a twist angle of 1.05 degrees, a strain angle of 15 degrees, a strain magnitude of 0.5%, and zero displacement field.

tdbg = TDBGModel(1.05, 15, 0.005, 0)
gen_ham(kx, ky, xi=1)
Generate hamiltonian for a given k-point.

Parameters:
kx (float) – x and y coordinates of the momentum point for which the Hamiltonian is to be generated in inverse nanometers. Note that kx points along Gamma-Gamma of the moire Brillouin zone.

ky (float) – x and y coordinates of the momentum point for which the Hamiltonian is to be generated in inverse nanometers. Note that kx points along Gamma-Gamma of the moire Brillouin zone.

xi (+/- 1) – Valley index.

Returns:
ham

Return type:
numpy matrix, shape (4*Nq, 4*Nq), dtype = complex

Examples

Generate Hamiltonian and solve for eigenstates at the K point of the moire Brillouin zone (kx=ky=0). Eigenvalues and eigenvectors are stored in vals and vecs respectively.

tdbg = TDBGModel(1.05, 15, 0.005, 0)
ham =  tdbg.gen_ham(0,0)
vals, vecs = eigh(ham)
solve_LDOS(nk=16, px=64, sz=40, l1=1, l2=1, energies=array([-0.1, -0.099, -0.098, -0.097, -0.096, -0.095, -0.094, -0.093, -0.092, -0.091, -0.09, -0.089, -0.088, -0.087, -0.086, -0.085, -0.084, -0.083, -0.082, -0.081, -0.08, -0.079, -0.078, -0.077, -0.076, -0.075, -0.074, -0.073, -0.072, -0.071, -0.07, -0.069, -0.068, -0.067, -0.066, -0.065, -0.064, -0.063, -0.062, -0.061, -0.06, -0.059, -0.058, -0.057, -0.056, -0.055, -0.054, -0.053, -0.052, -0.051, -0.05, -0.049, -0.048, -0.047, -0.046, -0.045, -0.044, -0.043, -0.042, -0.041, -0.04, -0.039, -0.038, -0.037, -0.036, -0.035, -0.034, -0.033, -0.032, -0.031, -0.03, -0.029, -0.028, -0.027, -0.026, -0.025, -0.024, -0.023, -0.022, -0.021, -0.02, -0.019, -0.018, -0.017, -0.016, -0.015, -0.014, -0.013, -0.012, -0.011, -0.01, -0.009, -0.008, -0.007, -0.006, -0.005, -0.004, -0.003, -0.002, -0.001, 0., 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.05, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069, 0.07, 0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079, 0.08, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089, 0.09, 0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099, 0.1]), xi=1)
Solves for the local density of states in real space. The computation is performed for a momentum space grid and then transformed to real space.

Parameters:
nk (int) – The total number of points in the moire Brillouin zone is nk**2. Larger nk corresponds to more accurate and more computationally expensive calculations.

px (int) – Number of pixels to include in the real space local density of states image. The local density of states is computed with a square aspect ratio and px is the number of pixels along one side of the square.

sz (float) – Size of the real space local density of states window. Generated images are sz x sz nm^2

l1 (float) – Weight of top layer and second layer respectively. Typically numbers between 0 and 1. Projecting onto the top layer only corresponds to l1=1, l2=0.

l2 (float) – Weight of top layer and second layer respectively. Typically numbers between 0 and 1. Projecting onto the top layer only corresponds to l1=1, l2=0.

energies (numpy array of floats) – A list of energies in electron-Volts at which to compute the local density of states.

+/-1 (xi =) – Valley index.

Returns:
m – The local density of states in real space computed at each input energy. m[:,:,i] is the local density of states over a sz x sz nm^2 window at energy energies[i].

Return type:
numpy array, shape (px, px, len(energies))

Examples

Compute the Fermi level (energy = 0 eV) local density of states over a 40 nm window with 256 pixel resolution.

tdbg = TDBGModel(1.05, 15, 0.005, 0)
m = tdbg.solve_LDOS(px = 256, sz = 40, energies = np.array([0]))
Visualize the result.

plt.imshow(m[:,:,0])
solve_PDOS(nk=16, energies=array([-0.1, -0.099, -0.098, -0.097, -0.096, -0.095, -0.094, -0.093, -0.092, -0.091, -0.09, -0.089, -0.088, -0.087, -0.086, -0.085, -0.084, -0.083, -0.082, -0.081, -0.08, -0.079, -0.078, -0.077, -0.076, -0.075, -0.074, -0.073, -0.072, -0.071, -0.07, -0.069, -0.068, -0.067, -0.066, -0.065, -0.064, -0.063, -0.062, -0.061, -0.06, -0.059, -0.058, -0.057, -0.056, -0.055, -0.054, -0.053, -0.052, -0.051, -0.05, -0.049, -0.048, -0.047, -0.046, -0.045, -0.044, -0.043, -0.042, -0.041, -0.04, -0.039, -0.038, -0.037, -0.036, -0.035, -0.034, -0.033, -0.032, -0.031, -0.03, -0.029, -0.028, -0.027, -0.026, -0.025, -0.024, -0.023, -0.022, -0.021, -0.02, -0.019, -0.018, -0.017, -0.016, -0.015, -0.014, -0.013, -0.012, -0.011, -0.01, -0.009, -0.008, -0.007, -0.006, -0.005, -0.004, -0.003, -0.002, -0.001, 0., 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.05, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069, 0.07, 0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079, 0.08, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089, 0.09, 0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099, 0.1]), xi=1, plot_it=True)
Solves the Hamiltonian for a grid of k points within the first Brillouin zone and computes the density of states (i.e. momentum integrated number of solutions per unit energy) projected onto each of the four layers.

Parameters:
nk (int) – The total number of points in the moire Brillouin zone is nk**2. Larger nk corresponds to more accurate and more computationally expensive calculations.

energies (numpy array of floats) – A list of energies in electron-Volts at which to compute the density of states.

xi (+/-1) – Valley index.

plot_it (boolean) – If true a function call will display a plot of the density of states projected onto each layer.

Returns:
PDOS – The list contains a numpy array containing the energy dependent density of states for each of the four layers (elements 0-3) and the total density of states across all layers (element 4). PDOS[l][i] is the density of states on layer l at energy energies[i].

Return type:
list of five numpy arrays, each of shape (len(energies),)

Examples

Generate and plot the density of states on the top layer for energies at 2 meV increments from -100 to +100 meV.

tdbg = TDBGModel(1.05, 15, 0.005, 0)
x = np.round(np.arange(-0.1,0.102,0.002),3)
PDOS = tdbg.solve_PDOS(energies = x)
plt.plot(x,PDOS[0])
Plot the density of states on the second layer from the top.

plt.plot(x,PDOS[1])
Plot the full density of states.

plt.plot(x,PDOS[4])
solve_along_path(res=16, plot_it=True, return_eigenvectors=False)
Compute the band structure along a path within the moire Brillouin zone. The path is predefined to be K -> Gamma -> M -> K’.

Parameters:
res (float) – The resolution of the cut, defined as the number of points to sample along the line Gamma -> K

plot_it (boolean) – If true a function call will display a plot of the resulting band structure.

return_eigenvectors (boolean) – If true return eigenvectors along the k path.

Returns:
evals_m (numpy array, shape (len(kpath), 4*Nq)) – Eigenvalues along the k path for valley xi = -1. Axis 0 indexes points along the k path. Axis 1 indexes bands.

evals_p (numpy array, shape (len(kpath), 4*Nq)) – Eigenvalues along the k path for valley xi = +1. Axis 0 indexes points along the k path. Axis 1 indexes bands.

evecs_m (numpy array, shape (len(kpath), 4*Nq, 4*Nq)) – Eigenvectors along the k path for valley xi = -1. Axis 0 indexes points along the k path. Axis 1 indexes into the eigenvector. Axis 2 indexes bands. E.g. evecs_m[i,:,j] is the eigenvector for k point i and band j.

evecs_p (numpy array, shape (len(kpath), 4*Nq, 4*Nq)) – Eigenvectors along the k path for valley xi = +1. Axis 0 indexes points along the k path. Axis 1 indexes into the eigenvector. Axis 2 indexes bands. E.g. evecs_p[i,:,j] is the eigenvector for k point i and band j.

kpath (list of shape (2,) numpy arrays) – List of k points along the path. Each point is represented as a two component array in the list.

Examples

Solve for the bands along the path K -> Gamma -> M -> K’, returning eigenvectors.

tdbg = TDBGModel(1.05, 15, 0.005, 0)
evals_m, evals_p, evecs_m, evecs_p, kpath = tdbg.solve_along_path(return_eigenvectors = True)
The same thing without returning eigenvectors.

evals_m, evals_p, kpath = tdbg.solve_along_path()
A higher resolution computation.

evals_m, evals_p, kpath = tdbg.solve_along_path(res = 64)
class pytwist.TTGModel(theta, phi, epsilon, D, a=0.246, beta=3.14, delta=0.16, vf=1, u=0.087, up=0.105, cut=4)
Bases: object

This is the model class for mirror symmetric twisted trilayer graphene in AtA configuration.

See S. Turkel et al. Science 376 (6589), 193-199.

thetafloat
Twist angle between the top and bottom bilayer in degrees.

phifloat
Heterostrain angle in degrees relative to the atomic lattice.

epsilonfloat
Heterostrain magnitude. For example, a value of 0.01 corresponds to a 1% difference in lattice constants between the top and bottom layers.

Dfloat
Potential difference between top and bottom layers in Volts. This captures the effect of an external displacement field.

afloat
Lattice constant of graphene in nanometers.

betafloat
Two center hopping modulus. See Phys. Rev. B 100, 035448 (2019).

deltafloat
Poisson ratio for graphene.

vffloat
Fermi velocity renormalization constant. Typical values are 1.2 - 1.3.

u, upfloat
Interlayer hopping amplitudes in electron-Volts for AA and AB sites respectively. u != up captures the effect of finite out of plane corrugation of the moire lattice.

cutfloat
Momentum space cut off in units of the moire lattice vector. Larger values will result in a larger Hamiltonian matrix. Convergence is typically attained by cut = 4.

Examples

Construct a model object representing a twisted trilayer graphene device with a twist angle of 1.56 degrees, a strain angle of 15 degrees, a strain magnitude of 0.5%, and zero displacement field.

ttg = TTGModel(1.56, 15, 0.005, 0)
gen_ham(kx, ky, xi=1)
Generate hamiltonian for a given k-point.

Parameters:
kx (float) – x and y coordinates of the momentum point for which the Hamiltonian is to be generated in inverse nanometers. Note that kx points along Gamma-Gamma of the moire Brillouin zone.

ky (float) – x and y coordinates of the momentum point for which the Hamiltonian is to be generated in inverse nanometers. Note that kx points along Gamma-Gamma of the moire Brillouin zone.

xi (+/- 1) – Valley index.

Returns:
ham

Return type:
numpy matrix, shape (2*Nq, 2*Nq), dtype = complex

Examples

Generate Hamiltonian and solve for eigenstates at the K point of the moire Brillouin zone (kx=ky=0). Eigenvalues and eigenvectors are stored in vals and vecs respectively.

ttg = TTGModel(1.56, 15, 0.005, 0)
ham =  ttg.gen_ham(0,0)
vals, vecs = eigh(ham)
solve_PDOS(nk=16, energies=array([-0.1, -0.099, -0.098, -0.097, -0.096, -0.095, -0.094, -0.093, -0.092, -0.091, -0.09, -0.089, -0.088, -0.087, -0.086, -0.085, -0.084, -0.083, -0.082, -0.081, -0.08, -0.079, -0.078, -0.077, -0.076, -0.075, -0.074, -0.073, -0.072, -0.071, -0.07, -0.069, -0.068, -0.067, -0.066, -0.065, -0.064, -0.063, -0.062, -0.061, -0.06, -0.059, -0.058, -0.057, -0.056, -0.055, -0.054, -0.053, -0.052, -0.051, -0.05, -0.049, -0.048, -0.047, -0.046, -0.045, -0.044, -0.043, -0.042, -0.041, -0.04, -0.039, -0.038, -0.037, -0.036, -0.035, -0.034, -0.033, -0.032, -0.031, -0.03, -0.029, -0.028, -0.027, -0.026, -0.025, -0.024, -0.023, -0.022, -0.021, -0.02, -0.019, -0.018, -0.017, -0.016, -0.015, -0.014, -0.013, -0.012, -0.011, -0.01, -0.009, -0.008, -0.007, -0.006, -0.005, -0.004, -0.003, -0.002, -0.001, 0., 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.03, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.04, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.05, 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.06, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068, 0.069, 0.07, 0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079, 0.08, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089, 0.09, 0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099, 0.1]), xi=1, plot_it=True)
Solves the Hamiltonian for a grid of k points within the first Brillouin zone and computes the density of states (i.e. momentum integrated number of solutions per unit energy) projected onto each of the four layers.

Parameters:
nk (int) – The total number of points in the moire Brillouin zone is nk**2. Larger nk corresponds to more accurate and more computationally expensive calculations.

energies (numpy array of floats) – A list of energies in electron-Volts at which to compute the density of states.

xi (+/-1) – Valley index.

plot_it (boolean) – If true a function call will display a plot of the density of states projected onto each layer.

Returns:
PDOS – The list contains a numpy array containing the energy dependent density of states for each of the three layers (elements 0-2) and the total density of states across all layers (element 3). PDOS[l][i] is the density of states on layer l at energy energies[i].

Return type:
list of three numpy arrays, each of shape (len(energies),)

Examples

Generate and plot the density of states on the top layer for energies at 2 meV increments from -100 to +100 meV.

ttg = TTGModel(1.56, 15, 0.005, 0)
x = np.round(np.arange(-0.1,0.102,0.002),3)
PDOS = ttg.solve_PDOS(energies = x)
plt.plot(x,PDOS[0])
Plot the density of states on the second layer from the top.

plt.plot(x,PDOS[1])
Plot the full density of states.

plt.plot(x,PDOS[3])
solve_along_path(res=16, plot_it=True, return_eigenvectors=False)
Compute the band structure along a path within the moire Brillouin zone. The path is predefined to be K’ -> K -> Gamma -> M -> K’. Note that this is different from the TDBG class so as to clearly display the Dirac crossing.

Parameters:
res (float) – The resolution of the cut, defined as the number of points to sample along the line Gamma -> K

plot_it (boolean) – If true a function call will display a plot of the resulting band structure.

return_eigenvectors (boolean) – If true return eigenvectors along the k path.

Returns:
evals_m (numpy array, shape (len(kpath), 2*Nq)) – Eigenvalues along the k path for valley xi = -1. Axis 0 indexes points along the k path. Axis 1 indexes bands.

evals_p (numpy array, shape (len(kpath), 2*Nq)) – Eigenvalues along the k path for valley xi = +1. Axis 0 indexes points along the k path. Axis 1 indexes bands.

evecs_m (numpy array, shape (len(kpath), 2*Nq, 2*Nq)) – Eigenvectors along the k path for valley xi = -1. Axis 0 indexes points along the k path. Axis 1 indexes into the eigenvector. Axis 2 indexes bands. E.g. evecs_m[i,:,j] is the eigenvector for k point i and band j.

evecs_p (numpy array, shape (len(kpath), 2*Nq, 2*Nq)) – Eigenvectors along the k path for valley xi = +1. Axis 0 indexes points along the k path. Axis 1 indexes into the eigenvector. Axis 2 indexes bands. E.g. evecs_p[i,:,j] is the eigenvector for k point i and band j.

kpath (list of shape (2,) numpy arrays) – List of k points along the path. Each point is represented as a two component array in the list.

Examples

Solve for the bands along the path K’ -> K -> Gamma -> M -> K’, returning eigenvectors.

ttg = TTGModel(1.56, 15, 0.005, 0)
evals_m, evals_p, evecs_m, evecs_p, kpath = ttg.solve_along_path(return_eigenvectors = True)
The same thing without returning eigenvectors.

evals_m, evals_p, kpath = ttg.solve_along_path()
A higher resolution computation.

evals_m, evals_p, kpath = ttg.solve_along_path(res = 64)
pytwist
Navigation
Quick search
©2022, Simon Turkel. | Powered by Sphinx 5.1.1 & Alabaster 0.7.12 | Page source
