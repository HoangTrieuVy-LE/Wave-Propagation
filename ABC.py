import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class Grid1D:
	def __init__(self, left, right, npoints):
		self.left  = left
		self.right = right
		self.npoints = npoints
		self.grid_size = (right - left) / (npoints - 1)
		self.points  = np.linspace(left, right, npoints)

class BoundaryCondition(Enum):
	Dirichlet    = 0
	Neumann      = 1
	Periodic     = 2
	Transparent  = 3
	Approximate  = 4
 
def WaveLeapFrog(grid, gamma, c, T, U0, U1, boundary_condition):

	#for animation
	fig   = plt.figure()
	line, = plt.plot(grid.points, U0)
	line.set_ydata(U1)
	
	plt.title("{} Boundary Condition".format(boundary_condition.name))
	plt.xlim(grid.left, grid.right)
	plt.ylim(1.1 * max(U0), -max(U0))
	plt.gca().invert_yaxis()

	assert gamma/c < 1.0, "CFL must be less than 1."
	dx = grid.grid_size
	dt = gamma**2 / c**2 * dx

	Uprev = U0                      #Uprev <-> U^{n-1}
	Ucurr = U1                      #Ucurr <-> U^n
	Unext = np.empty(grid.npoints)  #Unext <-> U^{n+1}
	
	left_values  = [U0[1] , U1[1] ]
	right_values = [U0[-2], U1[-2]]
	
	rminus = [0, gamma**2]
	rplus  = [2 - 2 / gamma**2, 1 / gamma**2 - gamma**2]

	t = dt 

	#main loop
	while (t <= T):
		Unext[1:-1] = 2 * Ucurr[1:-1] - Uprev[1:-1] + gamma**2 * (Ucurr[2:] - 2 * Ucurr[1:-1] + Ucurr[:-2])
		
		if (boundary_condition == BoundaryCondition.Dirichlet):
			Unext[0]  = 0
			Unext[-1] = 0
		elif (boundary_condition == BoundaryCondition.Neumann):
			Unext[0]  = Unext[1]
			Unext[-1] = Unext[-2]
		elif (boundary_condition == BoundaryCondition.Periodic):
			Unext[0]  = Unext[-2]
			Unext[-1] = Unext[1]
		elif (boundary_condition == BoundaryCondition.Transparent):
			left_values  += [ Unext[1 ]                                 ]
			right_values += [ Unext[-2]                                 ]
			rminus       += [ -gamma**2 * np.vdot(rplus, rminus[::-1])  ]
			rplus        += [ -rminus[-1]                               ]
			Unext[-1]    = np.vdot(rminus, right_values[::-1])
			Unext[0 ]    = np.vdot(rminus, left_values[::-1])
		elif (boundary_condition == BoundaryCondition.Approximate):
			pass
		else:
			raise ValueError("Boundary condition not available.")
		
		Uprev = np.copy(Ucurr)
		Ucurr = np.copy(Unext)
		
		t += dt
		line.set_ydata(Ucurr)
		plt.pause(0.005)
		
	plt.show()
  

grid = Grid1D(left = 0.0, right = 1.0, npoints = 500)
U0  = np.exp(-100 * (grid.points - 0.5 * (grid.left + grid.right))**2) #Initial condtion 
U1   = np.copy(U0) #Obtained from the null initial velocity

gamma = 0.9
c     = 1.0
T     = 1.5

WaveLeapFrog(grid, gamma, c, T, U0, U1, BoundaryCondition.Transparent)
	
	
