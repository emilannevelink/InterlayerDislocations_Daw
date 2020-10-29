
import numpy as np
#import scipy.special as sp
#import scipy.optimize as opt
from scipy.interpolate import griddata
import pdb, os, copy
import matplotlib.pyplot as plt
#from scipy import optimize
from mpl_toolkits.mplot3d import axes3d


import _IDDAW as cfiles

"""
Dislocationinformation: list of dislocations
each dislocation: location, line direction, burgersvector
=> 3x3 numpy array
location is the axis intercept
"""

#Constants
a= 1.42#*3**0.5
rt3 = 3**0.5

def createCijkl(E=18.5,mu = 5.49):
	"""
	E = 16.93#*100
	G = 6.9
	
	lam = G*(E-2*G)/(3*G-E)

	C11 = lam+2*G
	C12 = lam
	C66 = G
	"""
	#Rebo Single layer - Research Update 6/18
	#C12 = 7.15
	C11 = E
	C66 = mu#5.49
	#C11 = C12+2*C66
	C12 = C11-2*C66

	C13 = 0#0.026
	C33 = 0#0.64
	C44 = 0#.011 #Check
	
	C14 = 0#0.002
	C15 = 0#-0.002
	
	#C34;C35;C36;C16;C26;C45 are all symmetrically zero
	C34 = 0#.08

	Cijkl = np.zeros((3,3,3,3),dtype='complex128')
	#C11;C22
	Cijkl[0,0,0,0], Cijkl[1,1,1,1] = C11*np.ones(2)
	#C12
	Cijkl[0,0,1,1], Cijkl[1,1,0,0] = C12*np.ones(2)
	#C66
	Cijkl[0,1,0,1], Cijkl[1,0,1,0], Cijkl[0,1,1,0], Cijkl[1,0,0,1] = C66*np.ones(4)
	#C33
	Cijkl[2,2,2,2] =  C33
	#C13; C23
	Cijkl[2,2,1,1], Cijkl[2,2,0,0],	Cijkl[0,0,2,2], Cijkl[1,1,2,2] = C13*np.ones(4)
	#C44
	Cijkl[2,1,2,1], Cijkl[1,2,1,2], Cijkl[2,1,1,2], Cijkl[1,2,2,1] = C44*np.ones(4)	
	#C55
	Cijkl[0,2,0,2], Cijkl[2,0,2,0], Cijkl[0,2,2,0], Cijkl[2,0,0,2] = C44*np.ones(4)	
	#C14
	Cijkl[2,1,0,0], Cijkl[1,2,0,0], Cijkl[0,0,2,1], Cijkl[0,0,1,2] = C14*np.ones(4)
	#C24
	Cijkl[2,1,1,1], Cijkl[1,2,1,1], Cijkl[1,1,2,1], Cijkl[1,1,1,2] = -C14*np.ones(4)
	#C56 - 1312
	Cijkl[0,2,0,1], Cijkl[0,2,1,0], Cijkl[2,0,0,1], Cijkl[2,0,1,0] = C14*np.ones(4)
	Cijkl[0,1,0,2], Cijkl[1,0,0,2], Cijkl[0,1,2,0], Cijkl[1,0,2,0] = C14*np.ones(4)
	
	#C15
	Cijkl[2,0,0,0], Cijkl[0,2,0,0], Cijkl[0,0,2,0], Cijkl[0,0,0,2] = C15*np.ones(4)
	#C25
	Cijkl[2,0,1,1], Cijkl[0,2,1,1], Cijkl[1,1,2,0], Cijkl[1,1,0,2] = -C15*np.ones(4)
	#C46
	Cijkl[1,2,0,1], Cijkl[2,1,0,1], Cijkl[1,2,1,0], Cijkl[2,1,1,0] = -C15*np.ones(4)
	Cijkl[0,1,1,2], Cijkl[0,1,2,1], Cijkl[1,0,1,2], Cijkl[1,0,2,1] = -C15*np.ones(4)

	#C34
	#Cijkl[1,2,2,2], Cijkl[2,1,2,2], Cijkl[2,2,1,2], Cijkl[2,2,2,1] = C34*np.ones(4)
	#C35
	Cijkl[0,2,2,2], Cijkl[2,0,2,2], Cijkl[2,2,0,2], Cijkl[2,2,2,0] = C34*np.ones(4)

	return Cijkl

def cret_Duftot(fG,nlayers,Cijkl,kap,a1,a2,pmax,qmax,Lxy,DI_in,rcin,	DI_out,M,c33,alpha,beta,z0,xdata,ydata,raw=False,split=True):

	X = np.array([xdata,ydata]).transpose()

	ncoor = len(X)
	if not 2*ncoor == len(fG):
		print('Your fourier transformation will be wrong')
		pdb.set_trace()
		
	if nlayers == 2:
		fG1, fG2 = fG.reshape((2,len(fG)/2))

	C = np.concatenate([Cijkl[:2,:2,:2,:2].flatten()[::2],Cijkl[:2,:2,:2,:2].flatten()[1::2]],axis=0).reshape(2,8)

	C = C.real

	if len(DI_in)>0:
		locin = np.array(DI_in)[:,0,:2]
		burgersin = np.array(DI_in)[:,1,:2]
		ndislin = np.array(DI_in).shape[0]
	else:
		locin = []
		burgersin = []
		ndislin = 0

	locout = np.array(DI_out)[:,0,:2]
	dirout = np.array(DI_out)[:,1,:2]
	burgersout = np.array(DI_out)[:,2,:2]
	ndislout = np.array(DI_out).shape[0]
	
	rcout = np.outer(np.ones(len(DI_out)),[50,50]).flatten()
	res_cvisD = cfiles._visDuftot(C,Lxy,a1,a2,locin,burgersin,rcin,ndislin,fG1,fG2,pmax,qmax,locout,dirout,burgersout,rcout,ndislout,kap,c33,z0,M,ncoor,X,alpha)
		
	D = res_cvisD.reshape(len(X),8)
	D1_tot = D[:,:4].reshape(len(X),2,2)
	D2_tot = D[:,4:].reshape(len(X),2,2)

	if raw:
		return D1_tot, D2_tot
	else:
		return D1_tot-D2_tot

def cret_uftot(fG,nlayers,Cijkl,kap,a1,a2,pmax,qmax,Lxy,DI_in,rcin,	DI_out,M,c33,alpha,beta,z0,xdata,ydata,raw=False,split=False):
	
	X = np.array([xdata,ydata]).transpose()

	ncoor = len(X)
	if nlayers == 2:
		fG1, fG2 = fG.reshape((2,len(fG)/2))

	C = np.concatenate([Cijkl[:2,:2,:2,:2].flatten()[::2],Cijkl[:2,:2,:2,:2].flatten()[1::2]],axis=0).reshape(2,8)

	C = C.real

	if len(DI_in)>0:
		locin = np.array(DI_in)[:,0,:2]
		burgersin = np.array(DI_in)[:,1,:2]
		ndislin = np.array(DI_in).shape[0]
	else:
		locin = []
		burgersin = []
		ndislin = 0
	
	locout = np.array(DI_out)[:,0,:2]
	dirout = np.array(DI_out)[:,1,:2]
	burgersout = np.array(DI_out)[:,2,:2]
	ndislout = np.array(DI_out).shape[0]

	rcout = np.outer(np.ones(len(DI_out)),[50,50]).flatten()
	res_cvisu = cfiles._visuftot(C,Lxy,a1,a2,locin,burgersin,rcin,ndislin,fG1,fG2,pmax,qmax,locout,dirout,burgersout,rcout,ndislout,kap,c33,z0,M,ncoor,X,alpha)
	
	u = res_cvisu.reshape(len(X),4)
	#np.savetxt('u_Turk.txt', u)
	u1 = u[:,:2]
	u2 = u[:,2:]
	#top-bottom
	"""
	plt.quiver(xdata,ydata,u1[:,0],u1[:,1],scale_units='x',scale=.1,width=.01)
	#plt.show()
	plt.figure()
	plt.quiver(xdata,ydata,u2[:,0],u2[:,1],scale_units='x',scale=.1,width=.01)
	plt.figure()
	plt.quiver(xdata,ydata,(u1-u2)[:,0],(u1-u2)[:,1],scale_units='x',scale=.1,width=.01)
	plt.show()
	"""

	if raw:
		return u1, u2
	else:
		du = u1-u2
		du -= du[0]
		return du

def cEnergyf_sp(fG,nlayers,Cijkl,kap,a1,a2,pmax,qmax,Lxy,DI_in,rcin,DI_out,M,c33,alpha,beta,z0,xdata,ydata,raw=False,split=False):
	
	X = np.array([xdata,ydata]).transpose()

	ncoor = len(X)
	if not 2*ncoor == len(fG):
		print('Your fourier transformation will be wrong')
		pdb.set_trace()
	if nlayers == 2:
		fG1, fG2 = fG.reshape((2,len(fG)/2))

	C = np.concatenate([Cijkl[:2,:2,:2,:2].flatten()[::2],Cijkl[:2,:2,:2,:2].flatten()[1::2]],axis=0).reshape(2,8)

	C = C.real

	if len(DI_in)>0:
		locin = np.array(DI_in)[:,0,:2]
		burgersin = np.array(DI_in)[:,1,:2]
		ndislin = np.array(DI_in).shape[0]
	else:
		locin = []
		burgersin = []
		ndislin = 0
	
	locout = np.array(DI_out)[:,0,:2]
	dirout = np.array(DI_out)[:,1,:2]
	burgersout = np.array(DI_out)[:,2,:2]
	ndislout = np.array(DI_out).shape[0]
	
	rcout = np.outer(np.ones(len(DI_out)),[50,50]).flatten()
	E_sp = cfiles._Energyf_sp(C,Lxy,a1,a2,locin,burgersin,rcin,ndislin,fG1,fG2,pmax,qmax,locout,dirout,burgersout,rcout,ndislout,kap,c33,z0,M,ncoor,X,alpha)

	
	return E_sp

def grid_points(X,xgrid,u):

	u1_grid = np.zeros((len(xgrid),2))
	u1_grid[:,0] = griddata((X[:,0], X[:,1]), u[:,0], (xgrid[:,0],xgrid[:,1]), method='linear')#,fill_value=0)
	u1_grid[:,1] = griddata((X[:,0], X[:,1]), u[:,1], (xgrid[:,0],xgrid[:,1]), method='linear')#,fill_value=0)
	
	return u1_grid

def save1ddisplacements():
	fl_sv = 'test1d.txt'

	a = 1.42
	Cijkl = createCijkl()
	kap = 1.38
	C11 = 18
	c33 = 5.382
	nlayers = 2
	
	Lx = 1000
	Ly = 5

	
	DI = []
	rc = .94	

	M = np.array([0,0,0])
	alpha = .00252
	beta = 0
	z0 = 3.4

	phi = 90*np.pi/180
	
	a1 = np.array([Lx,0,0])
	a2 = np.array([0,Ly,0])
	a3 = np.array([0,0,1])

	Lxy = [a1[0],a2[1]]

	# define grid.
	xi = np.linspace(0,a1[0],int(a1[0]/2)/2*2)[:-1]
	yi = np.linspace(0,a2[1],2)[:-1]
	xx, yy = np.meshgrid(xi,yi)
	xdata = xx.flatten()
	ydata = yy.flatten()
	X = np.array([xdata,ydata]).transpose()

	pmax = int(((len(xi))-1)/2)
	qmax = int(((len(yi))-1)/2)
	fGic = 1e-2*np.zeros(2*(2*pmax+1)*(2*qmax+1))
	fGic[len(fGic)/4] = 3.4

	DI_12 =  lambda x1, phi: [np.array([[x1,0,0],[0,a2[1],0],[a*np.sin(phi),a*np.cos(phi),0]])]
	
	upart_continuum = cret_uftot(fGic,nlayers,Cijkl,kap,a1,a2,pmax,qmax,Lxy,DI,rc,DI_12(Lx/4,3*Lx/4,phi),[Rcs,Rct],M,c33,alpha,beta,z0,xdata,ydata,split=True)
	

	vals = np.concatenate([X,upart_continuum],axis=1)
	np.savetxt(fl_sv,vals)

def save2ddisplacements(disl = '2d90'):
	fl_sv = 'test%s.txt'%(disl)

	a = 1.42
	Cijkl = createCijkl()
	kap = 1.38
	C11 = 18
	c33 = 5.382
	nlayers = 2
	
	DI = []
	rc = .94	

	M = np.array([0,0,0])
	alpha = .00252
	beta = 0
	z0 = 3.4
	Rct = 67#161.25
	Rcs = 36#91.25

	Lm = 142.743607
	#nxgrid = 145
	#nygrid = 249

	#pdb.set_trace()
	Lx = Lm
	Ly = rt3*Lm
	

	a1 = np.array([Lx,0,0])
	a2 = np.array([0,Ly,0])
	a3 = np.array([0,0,1])

	Lxy = [a1[0],a2[1]]

	# define grid.
	xi = np.linspace(0,a1[0],int(a1[0]/2)/2*2+1)
	yi = np.linspace(0,a2[1],int(a2[1]/2)/2*2+1)
	xx, yy = np.meshgrid(xi,yi)
	xdata = xx.flatten()
	ydata = yy.flatten()
	X = np.array([xdata,ydata]).transpose()

	pmax = int(((len(xi))-1)/2)
	qmax = int(((len(yi))-1)/2)
	fGic = 1e-2*np.zeros(2*(2*pmax+1)*(2*qmax+1))
	fGic[len(fGic)/4] = 3.4

	if disl == '2d0':
		
		DI_12 = lambda x1, y1, x2, y2, x3, y3, x4, y4: [np.array([[x1,y1,0],[-Lx,0,0],[-a,0,0]]),np.array([[x2,y2,0],[-Lx,-Ly,0],[-a/2.,-rt3*a/2.,0]]),np.array([[x3,y3,0],[-Lx,Ly,0],[-a/2.,rt3*a/2.,0]]),np.array([[x4,y4,0],[-Lx,0,0],[-a,0,0]])]

		#AB Centered
		#y1 = Ly/6.
		#y2 = 2*Ly/3.
		
		#AA Centered
		y1 = 0
		y2 = Ly/2.
		upart_continuum = cret_uftot(fGic,nlayers,Cijkl,kap,a1,a2,pmax,qmax,Lxy,DI,rc,DI_12(0,y1,0,y1,0,y1,0,y2),M,c33,alpha,beta,z0,xdata,ydata,split=True,raw = False)


	elif disl == '2d90':
		
		DI_12 = lambda x1, y1, x2, y2, x3, y3, x4, y4: [np.array([[x1,y1,0],[-Lx,0,0],[0,a,0]]),np.array([[x2,y2,0],[-Lx,Ly,0],[rt3*a/2.,a/2.,0]]),np.array([[x3,y3,0],[-Lx,-Ly,0],[-rt3*a/2.,a/2.,0]]),np.array([[x4,y4,0],[-Lx,0,0],[0,a,0]])]

		#AB Centered
		y1 = Ly/3.
		y2 = 5*Ly/6.

		upart_continuum = cret_uftot(fGic,nlayers,Cijkl,kap,a1,a2,pmax,qmax,Lxy,DI,rc,DI_12(0,y1,0,y1,0,y1,0,y2),M,c33,alpha,beta,z0,xdata,ydata,split=True,raw = False)


	vals = np.concatenate([X,upart_continuum],axis=1)
	np.savetxt(fl_sv,vals)
	#pdb.set_trace()	

def plot2dfromsave(disl = '2d90'):
	
	
	Lm = 142.743607
	nxgrid = 145#289#51#579#1441#

	fl_sv = 'test%s.txt'%(disl)
	vals = np.genfromtxt(fl_sv)

	X = vals[:,[0,1]]
	upart_continuum = vals[:,[2,3]]

	xis = np.linspace(0,X[:,0].max(),30)
	yis = np.linspace(0,X[:,1].max(),54)
	xxs, yys = np.meshgrid(xis,yis)
	xdatas = xxs.flatten()
	ydatas = yys.flatten()

	xgrid = np.array([xdatas,ydatas]).transpose()

	upart_continuum_grid = grid_points(X,xgrid,upart_continuum)
	
	z_min, z_max = 0, 3#np.nanmax(np.real(u_diff))

	
	plt.figure(figsize=(5,9))
	plt.quiver(xdatas,ydatas,upart_continuum_grid[:,0],upart_continuum_grid[:,1],scale_units='x',scale=.2,width=.01)
	
	plt.axis('equal');plt.show()

	
	pdb.set_trace()

def energy_1d():
	#Output the line energy components [E_elastic_1, E_elastic_2, E_interface]
	
	a = 1.42
	Cijkl = createCijkl()
	kap = 1.38
	C11 = 18
	c33 = 5.382
	nlayers = 2
	
	Lx = 1000
	Ly = 5

	
	DI = []
	rc = .94	

	M = np.array([0,0,0])
	alpha = .00252
	beta = 0
	z0 = 3.4

	phi = 0*np.pi/180
	
	a1 = np.array([Lx,0,0])
	a2 = np.array([0,Ly,0])
	a3 = np.array([0,0,1])

	Lxy = [a1[0],a2[1]]

	# define grid.
	xi = np.linspace(0,a1[0],int(a1[0]/2)/2*2)[:-1]
	yi = np.linspace(0,a2[1],2)[:-1]
	xx, yy = np.meshgrid(xi,yi)
	xdata = xx.flatten()
	ydata = yy.flatten()
	X = np.array([xdata,ydata]).transpose()

	pmax = int(((len(xi))-1)/2)
	qmax = int(((len(yi))-1)/2)
	fGic = 1e-2*np.zeros(2*(2*pmax+1)*(2*qmax+1))
	fGic[len(fGic)/4] = 3.4

	DI_12 =  lambda x1, phi: [np.array([[x1,0,0],[0,a2[1],0],[a*np.sin(phi),a*np.cos(phi),0]])]
	
	energy = cEnergyf_sp(fGic,nlayers,Cijkl,kap,a1,a2,pmax,qmax,Lxy,DI,rc,DI_12(Lx/4,phi),M,c33,alpha,beta,z0,xdata,ydata,split=True)

	return energy/Ly #Line energy
	
def Distortion_2d(disl='2d90'):

	a = 1.42
	Cijkl = createCijkl()
	kap = 1.38
	C11 = 18
	c33 = 5.382
	nlayers = 2
	
	DI = []
	rc = .94	

	M = np.array([0,0,0])
	alpha = .00252
	beta = 0
	z0 = 3.4
	Rct = 67#161.25
	Rcs = 36#91.25

	Lm = 142.743607
	#nxgrid = 145
	#nygrid = 249

	#pdb.set_trace()
	Lx = Lm
	Ly = rt3*Lm
	

	a1 = np.array([Lx,0,0])
	a2 = np.array([0,Ly,0])
	a3 = np.array([0,0,1])

	Lxy = [a1[0],a2[1]]

	# define grid.
	xi = np.linspace(0,a1[0],int(a1[0]/2)/2*2+1)
	yi = np.linspace(0,a2[1],int(a2[1]/2)/2*2+1)
	xx, yy = np.meshgrid(xi,yi)
	xdata = xx.flatten()
	ydata = yy.flatten()
	X = np.array([xdata,ydata]).transpose()

	pmax = int(((len(xi))-1)/2)
	qmax = int(((len(yi))-1)/2)
	fGic = 1e-2*np.zeros(2*(2*pmax+1)*(2*qmax+1))
	fGic[len(fGic)/4] = 3.4

	if disl == '2d0':
		
		DI_12 = lambda x1, y1, x2, y2, x3, y3, x4, y4: [np.array([[x1,y1,0],[-Lx,0,0],[-a,0,0]]),np.array([[x2,y2,0],[-Lx,-Ly,0],[-a/2.,-rt3*a/2.,0]]),np.array([[x3,y3,0],[-Lx,Ly,0],[-a/2.,rt3*a/2.,0]]),np.array([[x4,y4,0],[-Lx,0,0],[-a,0,0]])]

		#AB Centered
		#y1 = Ly/6.
		#y2 = 2*Ly/3.
		
		#AA Centered
		y1 = 0
		y2 = Ly/2.
		D_tot = cret_Duftot(fGic,nlayers,Cijkl,kap,a1,a2,pmax,qmax,Lxy,DI,rc,DI_12(0,y1,0,y1,0,y1,0,y2),M,c33,alpha,beta,z0,xdata,ydata,raw=False)

	elif disl == '2d90':
		
		DI_12 = lambda x1, y1, x2, y2, x3, y3, x4, y4: [np.array([[x1,y1,0],[-Lx,0,0],[0,a,0]]),np.array([[x2,y2,0],[-Lx,Ly,0],[rt3*a/2.,a/2.,0]]),np.array([[x3,y3,0],[-Lx,-Ly,0],[-rt3*a/2.,a/2.,0]]),np.array([[x4,y4,0],[-Lx,0,0],[0,a,0]])]

		#AB Centered
		y1 = Ly/3.
		y2 = 5*Ly/6.

		D_tot = cret_Duftot(fGic,nlayers,Cijkl,kap,a1,a2,pmax,qmax,Lxy,DI,rc,DI_12(0,y1,0,y1,0,y1,0,y2),M,c33,alpha,beta,z0,xdata,ydata,raw=False)
	
	
	
	plt.figure(figsize=(5,9))
	plt.quiver(xdata,ydata,D_tot[:,0,0],D_tot[:,0,1])#,scale_units='y',scale=.5,width=.01, headwidth=3, headlength=5)
	plt.title('x-derivatives')
	plt.axis('equal');#plt.show()
	
	plt.figure(figsize=(5,9))
	plt.quiver(xdata,ydata,D_tot[:,1,0],D_tot[:,1,1])#,scale_units='y',scale=.5,width=.01, headwidth=3, headlength=5)
	plt.title('y-derivatives')
	plt.axis('equal');plt.show()


if __name__ == "__main__":

	print(energy_1d().sum())
	#Distortion_2d()
	#save2ddisplacements()
	#plot2dfromsave()
