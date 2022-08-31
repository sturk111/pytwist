import numpy as np
from numpy.linalg import eigh
from numpy.linalg import norm
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------
#Helper Functions

#generic rotation matrix
def R(x): 
	r = np.array((
		[np.cos(x),-np.sin(x)],
		[np.sin(x),np.cos(x)]
		))
	return r

#2x2 Identity matrix
I=np.identity(2)

#strain and rotation matrix
def Et(theta, theta_s, e, delta):
	return R(-theta_s)@np.array(([e,0],[0,-delta*e]))@R(theta_s) + np.array(([0,-theta],[theta,0]))

#-----------------------------------------------------------------------------------------------

#Create the continuum model class
class TDBGModel():
	def __init__(self, theta, phi, epsilon, D, #Empirical parameters (must provide as input)
		a = 0.246, beta = 3.14, delta = 0.16, #Graphene parameters
		vf = 1, u = 0.0797, up = 0.0975, cut=4): #Continuum model parameters

		#Convert angles from degrees to radians
		theta = theta*np.pi/180
		phi = phi*np.pi/180

		#Empirical parameters
		self.theta = theta #twist angle
		self.phi = phi #strain angle
		self.epsilon = epsilon #strain percent
		self.D = D #displacement field

		#Graphene parameters
		self.a = a #lattice constant
		self.beta = beta #two center hopping modulus 
		self.delta = delta #poisson ratio
		self.A = np.sqrt(3)*self.beta/2/a #gauge connection

		#Continuum model parameters
		self.v = vf*2.1354*a #vf = 1.3 used in publications
		self.v3 = np.sqrt(3)*a*0.32/2
		self.v4 = np.sqrt(3)*a*0.044/2
		self.gamma1 = 0.4
		self.Dp = 0.05

		self.omega = np.exp(1j*2*np.pi/3)
		self.u = u
		self.up = up


		#Define the graphene lattice in momentum space
		k_d = 4*np.pi/3/a
		k1 = np.array([k_d,0])
		k2 = np.array([np.cos(2*np.pi/3)*k_d,np.sin(2*np.pi/3)*k_d])
		k3 = -np.array([np.cos(np.pi/3)*k_d,np.sin(np.pi/3)*k_d])

		#Generate the strained moire reciprocal lattice vectors
		q1 = Et(theta,phi,epsilon,delta)@k1
		q2 = Et(theta,phi,epsilon,delta)@k2
		q3 = Et(theta,phi,epsilon,delta)@k3
		q = np.array([q1,q2,q3]) #put them all in a single array
		self.q = q
		k_theta = np.max([norm(q1),norm(q2),norm(q3)]) #used to define the momentum space cutoff
		self.k_theta = k_theta

		#"the Q lattice" refers to a lattice of monolayer K points on which the continuum model is defined
		#basis vectors for the Q lattice
		b1 = q[1]-q[2]
		b2 = q[0]-q[2]
		b3 = q[1]-q[0]
		b = np.array([b1,b2,b3])
		self.b = b

		#generate the Q lattice
		Q = np.array([np.array(list([i,j,0]@b - l*q[0]) + [l]) for i in range(-100,100) for j in range(-100,100) for l in [0,1] if norm([i,j,0]@b - l*q[0]) <= np.sqrt(3)*k_theta*cut])
		self.Q = Q
		Nq = len(Q)
		self.Nq = Nq

		#nearest neighbors on the Q lattice
		self.Q_nn={}
		for i in range(Nq):
			self.Q_nn[i] = [[np.round(Q[:,:2],3).tolist().index(list(np.round(Q[i,:2]+q[j],3))),j] for j in range(len(q)) if list(np.round(Q[i,:2]+q[j],3)) in np.round(Q[:,:2],3).tolist()]

		#physical momenta corresponding to points in the Q lattice
		Q2G=np.array([[l,l] for l in Q[:,2]])*q[0]+Q[:,:2]
		self.G = Q2G[Q[:,2]==1]

	#A function to create the hamiltonian for a given point kx, ky
	def gen_ham(self,kx,ky,xi=1):
		k = np.array([kx,ky]) #2d momentum vector

		#Create moire hopping matrices for valley index xi
		U1 = np.array((
			[self.u,self.up],
			[self.up,self.u]))

		U2 = np.array((
			[self.u,self.up*self.omega**(-xi)],
			[self.up*self.omega**(xi),self.u]))

		U3 = np.array((
			[self.u,self.up*self.omega**(xi)],
			[self.up*self.omega**(-xi),self.u]))

		#Create and populate Hamiltonian matrix
		ham = np.matrix(np.zeros((4*self.Nq,4*self.Nq),dtype=complex))

		for i in range(self.Nq):
			t = self.Q[i,2]
			l = np.sign(2*t-1)
			M = Et(l*xi*self.theta/2,self.phi,l*xi*self.epsilon/2,self.delta)
			E = (M + M.T)/2
			exx = E[0,0]
			eyy = E[1,1]
			exy = E[0,1]
			
			kj = (I+M)@(k + self.Q[i,:2] + xi*self.A*np.array([exx - eyy, -2*exy]))

			km = xi*kj[0] - 1j*kj[1]
			kp = xi*kj[0] + 1j*kj[1]


			#Populate diagonal blocks
			ham[4*i:4*i+4,4*i:4*i+4] = np.array(([xi*(3/4-t)*self.D,-self.v*km,self.v4*km,self.v3*kp],
												 [0,self.Dp/2 + xi*(3/4-t)*self.D,self.gamma1,self.v4*km],
												 [0,0,self.Dp/2 + xi*(1/4-t)*self.D,-self.v*km],
												 [0,0,0,xi*(1/4-t)*self.D]))
			
			#Populate off-diagonal blocks
			nn = self.Q_nn[i]
			for neighbor in nn:
				j = neighbor[0]
				p = neighbor[1]
				ham[4*j+2:4*j+4,4*i:4*i+2] = (p==0)*U1 + (p==1)*U2 + (p==2)*U3

		return ham + ham.H 

	#A function to solve for the bands along the path: K -> Gamma -> M -> K'
	def solve_along_path(self, res=16, plot_it = True, return_eigenvectors = False): #res = number of points per unit length in k space
		l1 = int(res) #K->Gamma
		l2 = int(np.sqrt(3)*res/2) #Gamma->M
		l3 = int(res/2) #M->K'

		kpath = [] #K -> Gamma -> M -> K'
		for i in np.linspace(0,1,l1):
			kpath.append(i*(self.q[0]+self.q[1])) #K->Gamma
		for i in np.linspace(0,1,l2):
			kpath.append(self.q[0] + self.q[1] + i*(-self.q[0]/2 - self.q[1])) #Gamma->M
		for i in np.linspace(0,1,l3):
			kpath.append(self.q[0]/2 + i*self.q[0]/2) #M->K'

		evals_m = []
		evals_p = []
		if return_eigenvectors:
			evecs_m = []
			evecs_p = []

		for kpt in kpath: #for each kpt along the path
			ham_m = self.gen_ham(kpt[0],kpt[1],-1) #generate and solve a hamiltonian for each valley
			ham_p = self.gen_ham(kpt[0],kpt[1],1)

			val, vec = eigh(ham_m)
			evals_m.append(val)
			if return_eigenvectors:
				evecs_m.append(vec)

			val, vec = eigh(ham_p)
			evals_p.append(val)
			if return_eigenvectors:
				evecs_p.append(vec)

		evals_m = np.array(evals_m)
		evals_p = np.array(evals_p)

		if plot_it:
			plt.figure(1)
			plt.clf()
			for i in range(len(evals_m[1,:])):
				plt.plot(evals_m[:,i],linestyle='dashed')
				plt.plot(evals_p[:,i])

			plt.ylim(-0.07,0.07)
			plt.xticks([0,l1,l1+l2,l1+l2+l3],['K', r'$\Gamma$', 'M', 'K\''])
			plt.ylabel('Energy (eV)')
			plt.tight_layout()
			plt.show()

		if return_eigenvectors:
			evecs_m = np.array(evecs_m)
			evecs_p = np.array(evecs_p)
			return evals_m, evals_p, evecs_m, evecs_p, kpath

		else:
			return evals_m, evals_p, kpath

	#A function to solve for the partial DOS projected onto each layer
	#returns PDOS at energies specified by "energies" input vector (eV)
	def solve_PDOS(self, nk = 16, energies = np.round(np.linspace(-.1,.1,201),3), xi = 1, plot_it = True):

		#helper function to convert from dos dictionary (see below) to 1D vector
		def makePdos(D,orbs):
			y = np.zeros((len(energies),len(orbs)))
			for o in range(len(orbs)):
				for j in range(len(energies)):
					y[j,o] += D[orbs[o]].get(energies[j],0)
			return np.sum(y,1)/len(orbs)

		#Define a grid of k points
		kpts = np.array([i*self.b[0] - j*self.b[1] for i in np.linspace(0,1,nk,endpoint=False) for j in np.linspace(0,1,nk,endpoint=False)])

		t = np.array([val for pair in zip(self.Q[:,2],self.Q[:,2],self.Q[:,2],self.Q[:,2]) for val in pair])
		BL = xi*(2*t-1)

		#Create masks for each sublattice/layer degree of freedom (to project the DOS)
		looo = np.array(self.Nq*[1,0,0,0])
		oloo = np.array(self.Nq*[0,1,0,0])
		oolo = np.array(self.Nq*[0,0,1,0])
		oool = np.array(self.Nq*[0,0,0,1])

		A1 = (BL==-1)*((t==0)*looo + (t==1)*oool)
		B1 = (BL==-1)*((t==0)*oloo + (t==1)*oolo)
		A2 = (BL==-1)*((t==0)*oolo + (t==1)*oloo)
		B2 = (BL==-1)*((t==0)*oool + (t==1)*looo)
		A3 = (BL==1)*((t==0)*oool + (t==1)*looo)
		B3 = (BL==1)*((t==0)*oolo + (t==1)*oloo)
		A4 = (BL==1)*((t==0)*oloo + (t==1)*oolo)
		B4 = (BL==1)*((t==0)*looo + (t==1)*oool)

		#Store masks in dictionary
		M = {'A1':A1,'B1':B1,'A2':A2,'B2':B2,'A3':A3,'B3':B3,'A4':A4,'B4':B4}

		#Create dictionary to store partial DOS for each sublattice (A/B) and layer (1,2,3,4)
		dos = {'A1':{},'B1':{},'A2':{},'B2':{},'A3':{},'B3':{},'A4':{},'B4':{}}

		#Solve model on grid of k points, storing the eigenvector amplitudes for each sublattice/layer
		for kpt in kpts:
			ham = self.gen_ham(kpt[0],kpt[1],xi)
			vals, vecs = eigh(ham)

			for j in range(len(vals)):
				val = np.round(vals[j],3)
				vec = np.array(vecs[:,j])

				for s in M:
					dos[s][val] = dos[s].get(val,0) + np.sum(abs(vec[M[s]==1,0])**2)

		#create PDOS for each layer
		L1 = makePdos(dos,['A1','B1'])
		L2 = makePdos(dos,['A2','B2'])
		L3 = makePdos(dos,['A3','B3'])
		L4 = makePdos(dos,['A4','B4'])
		PDOS = [L1, L2, L3, L4, (L1+L2+L3+L4)/4]

		if plot_it:
			plt.figure(1)
			plt.clf()
			for i in range(len(PDOS)):
				plt.plot(energies,PDOS[i])

			plt.xlabel('Energy (eV)')
			plt.ylabel('DOS')
			plt.legend(['L1','L2','L3','L4','Full'])
			plt.tight_layout()
			plt.show()

		#return list of layer projected DOS and full DOS
		return PDOS

	#A function to solve for the local density of states
	#returns an array of 2D LDOS images at the specified energies
	def solve_LDOS(self, nk=16, px = 64, sz = 40, l1 = 1, l2 = 1, energies = np.round(np.linspace(-.1,.1,201),3), xi=1):
		#nk^2 = momentum space grid size
		#px = number of real space pixels
		#sz = size of image in nm
		#l1/l2 = layer weights for outer and inner layers respectively

		#helper function to create 2D image from Fourier components
		def im(energy, px, eiGr, rho_G, G):
			gap = np.ones(len(G))
			amps = np.real(np.sum(rho_G.get(energy,gap)*eiGr.T,1))
			amps = np.reshape(amps,(px,px))
			return amps

		energies = np.array(energies)
		G = self.G

		#sublattice/layer masks
		t = np.array([val for pair in zip(self.Q[:,2],self.Q[:,2],self.Q[:,2],self.Q[:,2]) for val in pair])

		looo = np.array(self.Nq*[1,0,0,0])
		oloo = np.array(self.Nq*[0,1,0,0])
		oolo = np.array(self.Nq*[0,0,1,0])
		oool = np.array(self.Nq*[0,0,0,1])

		#Define grids in momentum space and real space
		kpts = np.array([i*self.b[0] - j*self.b[1] for i in np.linspace(0,1,nk,endpoint=False) for j in np.linspace(0,1,nk,endpoint=False)])
		l_theta = 4*np.pi/self.k_theta/3
		xhat = np.array([1,0])
		yhat = np.array([0,1])
		rpts = np.array([(-sz/2 + i*sz/px)*xhat + (-sz/2 + j*sz/px)*yhat for i in range(px) for j in range(px)]) - (l_theta/8)*xhat

		#Phase matrix
		eiGr = np.exp(-1j*np.einsum('ij,kj',G, rpts))

		#Create a matrix mapping indices of g in G and g' in G to the index of g + g' in G
		GxGp = np.zeros((len(G),len(G)),dtype='int16')
		for g in range(len(G)):
			for gp in range(len(G)):
				if list(np.round(G[g] + G[gp],3)) in np.round(G,3).tolist():
					GxGp[g,gp] = np.round(G,3).tolist().index(list(np.round(G[g] + G[gp],3)))
				else:
					GxGp[g,gp] = -1

		#Create dictionary to store DOS
		dos = {}

		#Solve the model on a grid of kpts
		for kpt in kpts:
			ham = self.gen_ham(kpt[0], kpt[1], xi)
			values, vectors = eigh(ham)
			for j in range(len(values)):
				val = np.round(values[j],3)
				vec = np.array(vectors[:,j].flatten().tolist()[0] + [0]) #add a zero on the end to mask -1 in GxGp
				if val in dos:
					dos[val].append(vec)
				else:
					dos[val] = [vec]

		#Generate fourier components of the LDOS
		rho_G = {}

		for val in dos:
			if val > np.max(energies) or val < np.min(energies):
				continue

			rho_G[val] = np.zeros(len(G),dtype='complex')

			psiA1 = np.array(dos[val])[:,np.append(oolo*t==1,True)]
			psiB1 = np.array(dos[val])[:,np.append(oool*t==1,True)]
			psiA2 = np.array(dos[val])[:,np.append(looo*t==1,True)]
			psiB2 = np.array(dos[val])[:,np.append(oloo*t==1,True)]


			rho_G[val] += ( l1*(np.einsum('ijk,ik',psiA1[:,GxGp],np.conjugate(psiA1[:,:len(G)])) + np.einsum('ijk,ik',psiB1[:,GxGp],np.conjugate(psiB1[:,:len(G)]))) + 
			l2*(np.einsum('ijk,ik',psiA2[:,GxGp],np.conjugate(psiA2[:,:len(G)])) + np.einsum('ijk,ik',psiB2[:,GxGp],np.conjugate(psiB2[:,:len(G)]))) )


		#Generate real space LDOS images
		m = np.zeros((px,px,len(energies)))
		for i,en in enumerate(energies):
			m[:,:,i] = im(en, px, eiGr, rho_G, G)

		return m

class TTGModel():
	def __init__(self, theta, phi, epsilon, D, #Empirical parameters (must provide as input)
		a = 0.246, beta = 3.14, delta = 0.16, #Graphene parameters
		vf = 1, u = 0.087, up = 0.105, cut=4): #Continuum model parameters

		#Convert angles from degrees to radians
		theta = theta*np.pi/180
		phi = phi*np.pi/180

		#Empirical parameters
		self.theta = theta #twist angle
		self.phi = phi #strain angle
		self.epsilon = epsilon #strain percent
		self.D = D #displacement field

		#Graphene parameters
		self.a = a #lattice constant
		self.beta = beta #two center hopping modulus 
		self.delta = delta #poisson ratio
		self.A = np.sqrt(3)*self.beta/2/a #gauge connection

		#Continuum model parameters
		self.v = vf*2.1354*a #vf = 1.3 used in publications
		self.v3 = np.sqrt(3)*a*0.32/2
		self.v4 = np.sqrt(3)*a*0.044/2
		self.gamma1 = 0.4
		self.Dp = 0.05

		self.omega = np.exp(1j*2*np.pi/3)
		self.u = u
		self.up = up


		#Define the graphene lattice in momentum space
		k_d = 4*np.pi/3/a
		k1 = np.array([k_d,0])
		k2 = np.array([np.cos(2*np.pi/3)*k_d,np.sin(2*np.pi/3)*k_d])
		k3 = -np.array([np.cos(np.pi/3)*k_d,np.sin(np.pi/3)*k_d])

		#Generate the strained moire reciprocal lattice vectors
		q1 = Et(theta,phi,epsilon,delta)@k1
		q2 = Et(theta,phi,epsilon,delta)@k2
		q3 = Et(theta,phi,epsilon,delta)@k3
		q = np.array([q1,q2,q3]) #put them all in a single array
		self.q = q
		k_theta = np.max([norm(q1),norm(q2),norm(q3)]) #used to define the momentum space cutoff
		self.k_theta = k_theta

		#"the Q lattice" refers to a lattice of monolayer K points on which the continuum model is defined
		#basis vectors for the Q lattice
		b1 = q[1]-q[2]
		b2 = q[0]-q[2]
		b3 = q[1]-q[0]
		b = np.array([b1,b2,b3])
		self.b = b

		#generate the Q lattice
		Q = np.array([np.array(list([i,j,0]@b - l*q[0]) + [l]) for i in range(-100,100) for j in range(-100,100) for l in [0,1] if norm([i,j,0]@b + l*q[0]) <= np.sqrt(3)*k_theta*cut])
		Q1 = np.array([[x[0],x[1],1] for x in Q if x[2]==1])
		Q2 = np.array([[x[0],x[1],2] for x in Q if x[2]==1])
		Q0 = np.array([item for item in Q if item[2]==0])
		Q = np.concatenate((Q0,Q1,Q2))
		self.Q = Q
		Nq = len(Q)
		self.Nq = Nq

		#nearest neighbors on the Q lattice
		self.Q_nn={}
		for i in range(Nq):
			self.Q_nn[i] = [[np.round(Q[:,:2],3).tolist().index(list(np.round(Q[i,:2]+q[j],3))),j] for j in range(len(q)) if list(np.round(Q[i,:2]+q[j],3)) in np.round(Q[:,:2],3).tolist()]


	#A function to create the hamiltonian for a given point kx, ky
	def gen_ham(self,kx,ky,xi=1):
		k = np.array([kx,ky]) #2d momentum vector

		#Create moire hopping matrices for valley index xi
		U1 = np.array((
			[self.u,self.up],
			[self.up,self.u]))

		U2 = np.array((
			[self.u,self.up*self.omega**(-xi)],
			[self.up*self.omega**(xi),self.u]))

		U3 = np.array((
			[self.u,self.up*self.omega**(xi)],
			[self.up*self.omega**(-xi),self.u]))

		#Create and populate Hamiltonian matrix
		ham = np.matrix(np.zeros((2*self.Nq,2*self.Nq),dtype=complex))

		for i in range(self.Nq):
			t = self.Q[i,2]
			if t==0:
				l=-1
			if t==1:
				l=1
			if t==2:
				l=1
			M = Et(l*xi*self.theta/2,self.phi,l*xi*self.epsilon/2,self.delta)
			E = (M + M.T)/2
			exx = E[0,0]
			eyy = E[1,1]
			exy = E[0,1]
			
			kj = (I+M)@(k + self.Q[i,:2] + xi*self.A*np.array([exx - eyy, -2*exy]))

			km = xi*kj[0] - 1j*kj[1]

			#Populate diagonal blocks
			ham[2*i,2*i+1] = -self.v*km 
			#add displacement field to outer layer diagonals
			ham[2*i,2*i] = (t==1)*self.D/2 - (t==2)*self.D/2
			ham[2*i+1,2*i+1] = (t==1)*self.D/2 - (t==2)*self.D/2

			#Populate off-diagonal blocks
			nn = self.Q_nn[i]
			for neighbor in nn:
				j = neighbor[0]
				p = neighbor[1]
				if t==1 or t==2:
					ham[2*i:2*i+2,2*j:2*j+2] = (p==0)*U1 + (p==1)*U2 + (p==2)*U3
				if t==0:
					ham[2*i:2*i+2,2*j:2*j+2] = (p==0)*U1 + (p==1)*U2 + (p==2)*U3

		return ham + ham.H 

	#A function to solve for the bands along the path: K -> Gamma -> M -> K'
	def solve_along_path(self, res=16, plot_it = True, return_eigenvectors = False): #res = number of points per unit length in k space
		l1 = int(res)
		l2 = int(res)
		l3 = int(np.sqrt(3)*res/2)
		l4 = int(res/2)

		Kprimept = np.array([0,0])
		Kpt = self.q[0]
		Gammapt = self.q[0] + self.q[1]
		Mpt = self.q[0]/2


		kpath = [] #K' -> K -> Gamma -> M -> K'
		for i in np.linspace(0,1,l1):
			kpath.append(Kprimept + i*(Kpt-Kprimept)) #K'->K
		for i in np.linspace(0,1,l2):
			kpath.append(Kpt + i*(Gammapt-Kpt)) #K->Gamma
		for i in np.linspace(0,1,l3):
			kpath.append(Gammapt + i*(Mpt-Gammapt)) #Gamma->M
		for i in np.linspace(0,1,l4):
			kpath.append(Mpt +i*(Kprimept-Mpt)) #M->K'
		
		evals_m = []
		evals_p = []
		if return_eigenvectors:
			evecs_m = []
			evecs_p = []

		for kpt in kpath: #for each kpt along the path
			ham_m = self.gen_ham(kpt[0],kpt[1],-1) #generate and solve a hamiltonian for each valley
			ham_p = self.gen_ham(kpt[0],kpt[1],1)

			val, vec = eigh(ham_m)
			evals_m.append(val)
			if return_eigenvectors:
				evecs_m.append(vec)

			val, vec = eigh(ham_p)
			evals_p.append(val)
			if return_eigenvectors:
				evecs_p.append(vec)

		evals_m = np.array(evals_m)
		evals_p = np.array(evals_p)

		if plot_it:
			plt.figure(1)
			plt.clf()
			for i in range(len(evals_m[1,:])):
				plt.plot(evals_m[:,i],linestyle='dashed')
				plt.plot(evals_p[:,i])

			plt.ylim(-0.07,0.07)
			plt.xticks([0,l1,l1+l2,l1+l2+l3],['K', r'$\Gamma$', 'M', 'K\''])
			plt.ylabel('Energy (eV)')
			plt.tight_layout()
			plt.show()

		if return_eigenvectors:
			evecs_m = np.array(evecs_m)
			evecs_p = np.array(evecs_p)
			return evals_m, evals_p, evecs_m, evecs_p, kpath

		else:
			return evals_m, evals_p, kpath

	#A function to solve for the partial DOS projected onto each layer
	#returns PDOS at energies specified by "energies" input vector (eV)
	def solve_PDOS(self, nk = 16, energies = np.round(np.linspace(-.1,.1,201),3), xi = 1, plot_it = True):
		#Define a grid of k points
		kpts = np.array([i*self.b[0] - j*self.b[1] for i in np.linspace(0,1,nk,endpoint=False) for j in np.linspace(0,1,nk,endpoint=False)])

		evals=[]
		evecs=[]

		for kpt in kpts:
			ham = self.gen_ham(kpt[0],kpt[1],xi)
			val,vec = eigh(ham)
			evals.append(val) 
			evecs.append(vec)

		evals = np.array(evals)
		evecs = np.array(evecs)

		#Generate partial ldos for top layer
		dos1 = {}
		dos2 = {}
		dos3 = {}

		L1 = self.Q[:,2]==1
		L1 = np.array([[x,x] for x in L1]).flatten()

		L2 = self.Q[:,2]==0
		L2 = np.array([[x,x] for x in L2]).flatten()

		L3 = self.Q[:,2]==2
		L3 = np.array([[x,x] for x in L3]).flatten()



		for i in range(len(evals[:,1])):
			for j in range(len(evals[1,:])):
				energy = np.round(evals[i,j],3)
				if energy in dos1:
					dos1[energy] += np.sum(abs(evecs[i,L1,j])**2)
					dos2[energy] += np.sum(abs(evecs[i,L2,j])**2)
					dos3[energy] += np.sum(abs(evecs[i,L3,j])**2)
				else:
					dos1[energy] = np.sum(abs(evecs[i,L1,j])**2)
					dos2[energy] = np.sum(abs(evecs[i,L2,j])**2)
					dos3[energy] = np.sum(abs(evecs[i,L3,j])**2)


		spec = np.zeros((len(energies),3))
		for x in range(len(energies)):
			spec[x,0] = dos1.get(energies[x],0)
			spec[x,1] = dos2.get(energies[x],0)
			spec[x,2] = dos3.get(energies[x],0)
		PDOS = [spec[:,0],spec[:,1],spec[:,2],np.mean(spec,axis=1)]
		

		if plot_it:
			plt.figure(1)
			plt.clf()
			for i in range(len(PDOS)):
				plt.plot(energies,PDOS[i])

			plt.xlabel('Energy (eV)')
			plt.ylabel('DOS')
			plt.legend(['L1','L2','L3','Full'])
			plt.tight_layout()
			plt.show()

		#return list of layer projected DOS and full DOS
		return PDOS

class TBGModel():
	def __init__(self, theta, phi, epsilon, #Empirical parameters (must provide as input)
		a = 0.246, beta = 3.14, delta = 0.16, #Graphene parameters
		vf = 1, u = 0.11, up = 0.11, cut=4): #Continuum model parameters

		#Convert angles from degrees to radians
		theta = theta*np.pi/180
		phi = phi*np.pi/180

		#Empirical parameters
		self.theta = theta #twist angle
		self.phi = phi #strain angle
		self.epsilon = epsilon #strain percent

		#Graphene parameters
		self.a = a #lattice constant
		self.beta = beta #two center hopping modulus 
		self.delta = delta #poisson ratio
		self.A = np.sqrt(3)*self.beta/2/a #gauge connection

		#Continuum model parameters
		self.v = vf*2.1354*a #vf = 1.3 used in publications
		self.v3 = np.sqrt(3)*a*0.32/2
		self.v4 = np.sqrt(3)*a*0.044/2
		self.gamma1 = 0.4
		self.Dp = 0.05

		self.omega = np.exp(1j*2*np.pi/3)
		self.u = u
		self.up = up

		#Define the graphene lattice in momentum space
		k_d = 4*np.pi/3/a
		k1 = np.array([k_d,0])
		k2 = np.array([np.cos(2*np.pi/3)*k_d,np.sin(2*np.pi/3)*k_d])
		k3 = -np.array([np.cos(np.pi/3)*k_d,np.sin(np.pi/3)*k_d])

		#Generate the strained moire reciprocal lattice vectors
		q1 = Et(theta,phi,epsilon,delta)@k1
		q2 = Et(theta,phi,epsilon,delta)@k2
		q3 = Et(theta,phi,epsilon,delta)@k3
		q = np.array([q1,q2,q3]) #put them all in a single array
		self.q = q
		k_theta = np.max([norm(q1),norm(q2),norm(q3)]) #used to define the momentum space cutoff
		self.k_theta = k_theta

		#"the Q lattice" refers to a lattice of monolayer K points on which the continuum model is defined
		#basis vectors for the Q lattice
		b1 = q[1]-q[2]
		b2 = q[0]-q[2]
		b3 = q[1]-q[0]
		b = np.array([b1,b2,b3])
		self.b = b

		#generate the Q lattice
		Q = np.array([np.array(list([i,j,0]@b - l*q[0]) + [l]) for i in range(-100,100) for j in range(-100,100) for l in [0,1] if norm([i,j,0]@b - l*q[0]) <= np.sqrt(3)*k_theta*cut])
		self.Q = Q
		Nq = len(Q)
		self.Nq = Nq

		#nearest neighbors on the Q lattice
		self.Q_nn={}
		for i in range(Nq):
			self.Q_nn[i] = [[np.round(Q[:,:2],3).tolist().index(list(np.round(Q[i,:2]+q[j],3))),j] for j in range(len(q)) if list(np.round(Q[i,:2]+q[j],3)) in np.round(Q[:,:2],3).tolist()]

	#A function to create the hamiltonian for a given point kx, ky
	def gen_ham(self,kx,ky,xi=1):
		k = np.array([kx,ky]) #2d momentum vector

		#Create moire hopping matrices for valley index xi
		U1 = np.array((
			[self.u,self.up],
			[self.up,self.u]))

		U2 = np.array((
			[self.u,self.up*self.omega**(-xi)],
			[self.up*self.omega**(xi),self.u]))

		U3 = np.array((
			[self.u,self.up*self.omega**(xi)],
			[self.up*self.omega**(-xi),self.u]))

		#Create and populate Hamiltonian matrix
		ham = np.matrix(np.zeros((2*self.Nq,2*self.Nq),dtype=complex))

		for i in range(self.Nq):
			t = self.Q[i,2]
			l = np.sign(2*t-1)
			M = Et(l*xi*self.theta/2,self.phi,l*xi*self.epsilon/2,self.delta)
			E = (M + M.T)/2
			exx = E[0,0]
			eyy = E[1,1]
			exy = E[0,1]
			
			kj = (I+M)@(k + self.Q[i,:2] + xi*self.A*np.array([exx - eyy, -2*exy]))

			km = xi*kj[0] - 1j*kj[1]


			#Populate diagonal blocks
			ham[2*i,2*i+1] = -self.v*km

			#Populate off-diagonal blocks
			nn = self.Q_nn[i]
			for neighbor in nn:
				j = neighbor[0]
				p = neighbor[1]
				ham[2*i:2*i+2,2*j:2*j+2] = (p==0)*U1 + (p==1)*U2 + (p==2)*U3

		return ham + ham.H 

	#A function to solve for the bands along the path: K -> Gamma -> M -> K'
	def solve_along_path(self, res=16, plot_it = True, return_eigenvectors = False): #res = number of points per unit length in k space
		l1 = int(res) #K->Gamma
		l2 = int(np.sqrt(3)*res/2) #Gamma->M
		l3 = int(res/2) #M->K'

		kpath = [] #K -> Gamma -> M -> K'
		for i in np.linspace(0,1,l1):
			kpath.append(i*(self.q[0]+self.q[1])) #K->Gamma
		for i in np.linspace(0,1,l2):
			kpath.append(self.q[0] + self.q[1] + i*(-self.q[0]/2 - self.q[1])) #Gamma->M
		for i in np.linspace(0,1,l3):
			kpath.append(self.q[0]/2 + i*self.q[0]/2) #M->K'

		evals_m = []
		evals_p = []
		if return_eigenvectors:
			evecs_m = []
			evecs_p = []

		for kpt in kpath: #for each kpt along the path
			ham_m = self.gen_ham(kpt[0],kpt[1],-1) #generate and solve a hamiltonian for each valley
			ham_p = self.gen_ham(kpt[0],kpt[1],1)

			val, vec = eigh(ham_m)
			evals_m.append(val)
			if return_eigenvectors:
				evecs_m.append(vec)

			val, vec = eigh(ham_p)
			evals_p.append(val)
			if return_eigenvectors:
				evecs_p.append(vec)

		evals_m = np.array(evals_m)
		evals_p = np.array(evals_p)

		if plot_it:
			plt.figure(1)
			plt.clf()
			for i in range(len(evals_m[1,:])):
				plt.plot(evals_m[:,i],linestyle='dashed')
				plt.plot(evals_p[:,i])

			plt.ylim(-0.07,0.07)
			plt.xticks([0,l1,l1+l2,l1+l2+l3],['K', r'$\Gamma$', 'M', 'K\''])
			plt.ylabel('Energy (eV)')
			plt.tight_layout()
			plt.show()

		if return_eigenvectors:
			evecs_m = np.array(evecs_m)
			evecs_p = np.array(evecs_p)
			return evals_m, evals_p, evecs_m, evecs_p, kpath

		else:
			return evals_m, evals_p, kpath

	def solve_DOS(self, nk = 16, energies = np.round(np.linspace(-.1,.1,201),3), xi = 1, plot_it = True):
		#Define a grid of k points
		kpts = np.array([i*self.b[0] - j*self.b[1] for i in np.linspace(0,1,nk,endpoint=False) for j in np.linspace(0,1,nk,endpoint=False)])

		evals = []
		evecs = []

		for kpt in kpts:
			ham = self.gen_ham(kpt[0],kpt[1],xi)
			vals, vecs = eigh(ham)
			evals.append(vals)

		dos=np.histogram(np.array(evals).flatten(),len(energies),range=(energies[0],energies[-1]))[0]

		if plot_it:
			plt.figure(1)
			plt.clf()
			plt.plot(energies,dos)

			plt.xlabel('Energy (eV)')
			plt.ylabel('DOS')
			plt.tight_layout()
			plt.show()

		return dos







