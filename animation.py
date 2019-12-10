import	numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from enum import Enum
#%% =========================================================================

class Grid1D:
	def __init__(self, left, right, npoints):
		self.left      = left
		self.right     = right
		self.npoints   = npoints
		self.grid_size = (right - left) / (npoints - 1)
		self.points    = np.linspace(left, right, npoints)
#%% =========================================================================

class BoundaryCondition(Enum):
	Dirichlet    = 0
	Neumann      = 1
	Periodic     = 2
	Transparent  = 3
	Approximate  = 4

#%% =========================================================================

class WaveLeapFrog:
	def __init__(self, grid, gamma, T, U0, U1, boundary_condition):
		assert gamma < 1.0
		self.grid               = grid
		self.gamma              = gamma
		self.Uprev              = U0
		self.Ucurr              = U1 
		self.Unext              = np.empty(self.Ucurr.shape)
		self.boundary_condition = boundary_condition

		self.simulation_time    = T
		self.time_step          = self.gamma**2 * self.grid.grid_size
		self.time_elapsed       = self.time_step
		self.time_points        = np.arange(2 * self.time_step, self.simulation_time, self.time_step)

		self.left_values  = [U0[1] , U1[1] ]
		self.right_values = [U0[-2], U1[-2]]
	
		self.rminus = [0, self.gamma**2]
		self.rplus  = [2 - 2 / self.gamma**2, 1 / self.gamma**2 - self.gamma**2]
#%% =========================================================================

	def update(self):
		if (self.time_elapsed <= self.simulation_time):
			self.Unext[1:-1] = 2 * self.Ucurr[1:-1] - self.Uprev[1:-1] + self.gamma**2 * (self.Ucurr[2:] 
												- 2 * self.Ucurr[1:-1] + self.Ucurr[:-2])
		
			if (self.boundary_condition == BoundaryCondition.Dirichlet):
				self.Unext[0]  = 0
				self.Unext[-1] = 0
			elif (self.boundary_condition == BoundaryCondition.Neumann):
				self.Unext[0]  = self.Unext[1]
				self.Unext[-1] = self.Unext[-2]
			elif (self.boundary_condition == BoundaryCondition.Periodic):
				self.Unext[0]  = self.Unext[-2]
				self.Unext[-1] = self.Unext[1]
			elif (self.boundary_condition == BoundaryCondition.Transparent):
				self.left_values  += [ self.Unext[1 ]                            				]
				self.right_values += [ self.Unext[-2]                                 			]
				self.rminus       += [ -self.gamma**2 * np.vdot(self.rplus, self.rminus[::-1])  ]
				self.rplus        += [ -self.rminus[-1]                               			]
				self.Unext[-1]    = np.vdot(self.rminus, self.right_values[::-1])
				self.Unext[0 ]    = np.vdot(self.rminus, self.left_values[::-1])
			elif (self.boundary_condition == BoundaryCondition.Approximate):
				pass
			else:
				raise ValueError("Boundary condition not available.")

			self.time_elapsed     += self.time_step
			self.Uprev = np.copy(self.Ucurr)
			self.Ucurr = np.copy(self.Unext)
#%% =========================================================================


grid = Grid1D(left = 0.0, right = 1.0, npoints = 500)
U0   = np.exp(-100 * (grid.points - 0.5 * (grid.left + grid.right))**2) #Initial condtion 
U1   = np.copy(U0) #Obtained from the null initial velocity
#%% =========================================================================

wave = WaveLeapFrog(grid, 0.9, 1.5, U0, U1, BoundaryCondition.Dirichlet)

fig   = plt.figure()
ax    = fig.add_subplot(111, autoscale_on = False, xlim = (grid.left, grid.right), ylim = (1.05 * max(U0), -1.05 * max(U0)))
plt.gca().invert_yaxis()
plt.title("{} Boundary Condition".format(wave.boundary_condition.name))
time_text = ax.text(0.02, 0.95, '', transform = ax.transAxes)
line, = ax.plot([], [])
#%% =========================================================================


def InitAnimation():
    line.set_data([], [])
    return line, 

def AnimatePropagation(i):
    global wave
    wave.update()
    
    line.set_data(wave.grid.points, wave.Ucurr)
    time_text.set_text("Time Elapsed: %.2f s" % wave.time_elapsed)
    return line, time_text, 
#%% =========================================================================

propagation = animation.FuncAnimation(fig, AnimatePropagation, init_func = InitAnimation, frames = 300, interval = 20, blit = True)

#Save animation as mp4
#Writer = animation.writers["ffmpeg"]
#writer = Writer(fps = 60, bitrate = 1800)
#propagation.save("wave_propagtion.mp4", writer = writer)

plt.show()
#%% =========================================================================
