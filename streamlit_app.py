#=========================================================================================================================================================
# Author: Luis E C Rocha  - luis.rocha@ugent.be - Ghent University, Belgium  - 26.09.2022
#
# Description: This file contains the implementation of the Game of Life, including some pre-defined patterns
#              1. first install streamlit using "pip install streamlit" 
#              2. run the python code from the command prompt using "streamlit run week4_exercise1.py"  *streamlit does not work well with Jupyler notebook
#              3. when you run streamlit, it will open a tab in your default browser with the streamlit application *it works as a webpage hosted at the following URL:  Local URL: http://localhost:8501
#              3.1. therefore, you can either stop the application or refresh the webpage to "restart" the application 
#
#=========================================================================================================================================================

# Import essential modules/libraries
import numpy as np
import scipy.signal as sg
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from time import sleep

#===============================================================================================================
# THIS IS THE DEFINITION OF THE CLASS FOR THE MODEL
class GameOfLife:

    #===================================================================
    # This method initialises the system
    def __init__(self, N, pattern, boundary):

        # These are the parameters of the model, input by the user
        self.N = N
        self.pattern = pattern
        self.boundary = boundary

        # This is the size of the system - Ps: this little trick just makes sure the system (grid) will have integer length for a square grid
        NN = int(np.sqrt(self.N))**2
        self.grid_size = int(np.sqrt(NN))

#--------------------------------------------------------------------------------------------------
        # The arrays bellow are used to store the neighbours of each cell ("grid_neigh") and the current state of each cell ("grid")
        self.grid_neigh = np.zeros([self.grid_size, self.grid_size])
        self.grid = np.zeros([self.grid_size, self.grid_size])

        if self.pattern == "Block":
            self.pattern_1()
        elif self.pattern == "Beacon":
            self.pattern_2()
        elif self.pattern == "Glider":
            self.pattern_3()
        elif self.pattern == "Pulsar":
            self.pattern_4()
#--------------------------------------------------------------------------------------------------

    #====================================================================
    # MODEL: This method updates the state of the cells (grid) at each time step
    def run(self):

        #-----------------------------------------------------
        # Calls the method to calculate the number of neighbours given the chosen boundary conditions
        if self.boundary == "Periodic":
#            self.periodic_boundary()     # using the brute force method
            self.periodic_boundary_c()  # using the convolution method
        elif self.boundary == "Finite":
#            self.finite_boundary()       # using the brute force method
            self.finite_boundary_c()    # using the convolution method

#        print(self.grid_neigh)
        # Update the state of the cells following the game of life rules
        for i in range(0, self.grid_size):
            for j in range(0, self.grid_size):

                # Rule 1: A live cell with fewer than 2 live neighbours dies (loneliness)
                # first check if the cell is alive
                if self.grid[i,j] == 1:
                    if self.grid_neigh[i,j] < 2:
                        self.grid[i, j] = 0
                # Rule 2: A live cell with 2 or 3 live neighbours lives on to the next generation (happiness) | tip: if you organise well the sequence of IF statements, you can skip this if statement
                    elif (self.grid_neigh[i,j] == 2) | (self.grid_neigh[i, j] == 3):
                        self.grid[i, j] = 1
                # Rule 3: A live cell with more than 3 live neighbours dies (overpopulation)
                    elif self.grid_neigh[i, j] > 3:
                        self.grid[i, j] = 0
                # Rule 4: A dead cell with exactly 3 live neighbours becomes a live cell (reproduction)
                elif self.grid_neigh[i, j] == 3:
                    self.grid[i, j] = 1
        #-----------------------------------------------------


    #====================================================================
    # Define boundary conditions - periodic
    def periodic_boundary(self):

        # Count the number of neighbours
        
        # the 4 corners of the grid
        self.grid_neigh[0,0] = self.grid[self.grid_size-1,self.grid_size-1] + self.grid[self.grid_size-1,0] + self.grid[self.grid_size-1, 1] + self.grid[0,self.grid_size-1] + self.grid[0,1] + self.grid[1, self.grid_size-1] + self.grid[1,0] + self.grid[1, 1]
        self.grid_neigh[self.grid_size-1,0] = self.grid[self.grid_size-2,self.grid_size-1] + self.grid[self.grid_size-2,0] + self.grid[self.grid_size-2, 1] + self.grid[self.grid_size-1,self.grid_size-1] + self.grid[self.grid_size-1,1] + self.grid[0, self.grid_size-1] + self.grid[0,0] + self.grid[0, 1]
        self.grid_neigh[0,self.grid_size-1] = self.grid[self.grid_size-1,self.grid_size-2] + self.grid[self.grid_size-1,self.grid_size-1] + self.grid[self.grid_size-1, 0] + self.grid[0,self.grid_size-2] + self.grid[0,0] + self.grid[1, self.grid_size-2] + self.grid[1,self.grid_size-1] + self.grid[1, 0]
        self.grid_neigh[self.grid_size-1,self.grid_size-1] = self.grid[self.grid_size-2,self.grid_size-2] + self.grid[self.grid_size-2,self.grid_size-1] + self.grid[self.grid_size-2, 0] + self.grid[self.grid_size-1,self.grid_size-2] + self.grid[self.grid_size-1,0] + self.grid[0, self.grid_size-2] + self.grid[0,self.grid_size-1] + self.grid[0, 0]

        # edges of the grid, EXCEPT the corners
        for k in range(1, self.grid_size-1):
            self.grid_neigh[k,0] = self.grid[k-1,self.grid_size-1] + self.grid[k-1,0] + self.grid[k-1, 1] + self.grid[k,self.grid_size-1] + self.grid[k,1] + self.grid[k+1, self.grid_size-1] + self.grid[k+1,0] + self.grid[k+1, 1]
            self.grid_neigh[0,k] = self.grid[self.grid_size-1,k-1] + self.grid[self.grid_size-1,k] + self.grid[self.grid_size-1, k+1] + self.grid[0,k-1] + self.grid[0,k+1] + self.grid[1, k-1] + self.grid[1,k] + self.grid[1,k+1]
            self.grid_neigh[self.grid_size-1,k] = self.grid[self.grid_size-2,k-1] + self.grid[self.grid_size-2,k] + self.grid[self.grid_size-2, k+1] + self.grid[self.grid_size-1,k-1] + self.grid[self.grid_size-1,k+1] + self.grid[0,k-1] + self.grid[0,k] + self.grid[0, k+1]
            self.grid_neigh[k,self.grid_size-1] = self.grid[k-1,self.grid_size-2] + self.grid[k-1,self.grid_size-1] + self.grid[k-1,0] + self.grid[k,self.grid_size-2] + self.grid[k,0] + self.grid[k+1,self.grid_size-2] + self.grid[k+1,self.grid_size-1] + self.grid[k+1,0]

            # all OTHER cells of the grid
            for l in range(1, self.grid_size-1):
                self.grid_neigh[k,l] = self.grid[k-1, l-1] + self.grid[k-1, l] + self.grid[k-1, l+1] + self.grid[k, l-1] + self.grid[k, l+1] + self.grid[k+1, l-1] + self.grid[k+1, l] + self.grid[k+1, l+1]

    #====================================================================
    # Define boundary conditions - finite
    def finite_boundary(self):

        # Count the number of neighbours
        
        # the 4 corners of the grid
        self.grid_neigh[0,0] = self.grid[0,1] + self.grid[1,0] + self.grid[1, 1]
        self.grid_neigh[self.grid_size-1,0] = self.grid[self.grid_size-2,0] + self.grid[self.grid_size-2, 1] + self.grid[self.grid_size-1,1]
        self.grid_neigh[0,self.grid_size-1] = self.grid[0,self.grid_size-2] + self.grid[1, self.grid_size-2] + self.grid[1,self.grid_size-1]
        self.grid_neigh[self.grid_size-1,self.grid_size-1] = self.grid[self.grid_size-2,self.grid_size-2] + self.grid[self.grid_size-2,self.grid_size-1] + self.grid[self.grid_size-1,self.grid_size-2]

        # edges of the grid, EXCEPT the corners
        for k in range(1, self.grid_size-1):
            
            # edge with j = 0 fixed
            self.grid_neigh[k,0] = self.grid[k-1,0] + self.grid[k-1, 1] + self.grid[k,1] + self.grid[k+1,0] + self.grid[k+1, 1]
            # edge with i = 0 fixed
            self.grid_neigh[0,k] = self.grid[0,k-1] + self.grid[0,k+1] + self.grid[1, k-1] + self.grid[1,k] + self.grid[1,k+1]
            # edge with i = size_grid-1 fixed
            self.grid_neigh[self.grid_size-1,k] = self.grid[self.grid_size-2,k-1] + self.grid[self.grid_size-2,k] + self.grid[self.grid_size-2, k+1] + self.grid[self.grid_size-1,k-1] + self.grid[self.grid_size-1,k+1]
            # edge with j = size_grid-1 fixed
            self.grid_neigh[k,self.grid_size-1] = self.grid[k-1,self.grid_size-2] + self.grid[k-1,self.grid_size-1] + self.grid[k,self.grid_size-2] + self.grid[k+1,self.grid_size-2] + self.grid[k+1,self.grid_size-1]

            # all OTHER cells of the grid
            for l in range(1, self.grid_size-1):
                self.grid_neigh[k,l] = self.grid[k-1, l-1] + self.grid[k-1, l] + self.grid[k-1, l+1] + self.grid[k, l-1] + self.grid[k, l+1] + self.grid[k+1, l-1] + self.grid[k+1, l] + self.grid[k+1, l+1]

    #====================================================================
    # Define boundary conditions - periodic
    # Using the convolution method
    def periodic_boundary_c(self):
        
        # define the kernel for the Moore neighbourhood
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        self.grid_neigh = sg.convolve2d(self.grid == 1, kernel, mode='same', boundary='wrap')
        
    #====================================================================
    # Define boundary conditions - fixed
    # Using the convolution method
    def finite_boundary_c(self):

        # define the kernel for the Moore neighbourhood
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        self.grid_neigh = sg.convolve2d(self.grid == 1, kernel, mode='same', boundary='fill')
                
    #====================================================================
    # Define initial pattern - block: still pattern
    # It is located at the centre of the lattice
    def pattern_1(self):

        self.grid[int(self.grid_size/2)  , int(self.grid_size/2)  ] = 1
        self.grid[int(self.grid_size/2)+1, int(self.grid_size/2)  ] = 1
        self.grid[int(self.grid_size/2)  , int(self.grid_size/2)+1] = 1
        self.grid[int(self.grid_size/2)+1, int(self.grid_size/2)+1] = 1

    #====================================================================
    # Define initial pattern - Beacon: oscillator pattern
    # It is located at the centre of the lattice
    def pattern_2(self):

        self.grid[int(self.grid_size/2)-1, int(self.grid_size/2)-1] = 1
        self.grid[int(self.grid_size/2)  , int(self.grid_size/2)-1] = 1
        self.grid[int(self.grid_size/2)-1, int(self.grid_size/2)  ] = 1

        self.grid[int(self.grid_size/2)+1, int(self.grid_size/2)+2] = 1
        self.grid[int(self.grid_size/2)+2, int(self.grid_size/2)+2] = 1
        self.grid[int(self.grid_size/2)+2, int(self.grid_size/2)+1] = 1

    #====================================================================
    # Define initial pattern - Glider: moving pattern
    # It is located at the centre of the lattice
    def pattern_3(self):

        self.grid[int(self.grid_size/2)+1, int(self.grid_size/2)  ] = 1
        self.grid[int(self.grid_size/2)  , int(self.grid_size/2)-1] = 1
        self.grid[int(self.grid_size/2)-1, int(self.grid_size/2)+1] = 1
        self.grid[int(self.grid_size/2)-1, int(self.grid_size/2)  ] = 1
        self.grid[int(self.grid_size/2)-1, int(self.grid_size/2)-1] = 1

    #====================================================================
    # Define initial pattern - Pulsar: oscillating pattern with cycle 3: repeats the pattern each 3 steps
    # It is located at the centre of the lattice
    def pattern_4(self):

        self.grid[int(self.grid_size/2)+2, int(self.grid_size/2)+1] = 1
        self.grid[int(self.grid_size/2)+3, int(self.grid_size/2)+1] = 1
        self.grid[int(self.grid_size/2)+4, int(self.grid_size/2)+1] = 1
        self.grid[int(self.grid_size/2)+1, int(self.grid_size/2)+2] = 1
        self.grid[int(self.grid_size/2)+1, int(self.grid_size/2)+3] = 1
        self.grid[int(self.grid_size/2)+1, int(self.grid_size/2)+4] = 1
        self.grid[int(self.grid_size/2)+6, int(self.grid_size/2)+2] = 1
        self.grid[int(self.grid_size/2)+6, int(self.grid_size/2)+3] = 1
        self.grid[int(self.grid_size/2)+6, int(self.grid_size/2)+4] = 1
        self.grid[int(self.grid_size/2)+2, int(self.grid_size/2)+6] = 1
        self.grid[int(self.grid_size/2)+3, int(self.grid_size/2)+6] = 1
        self.grid[int(self.grid_size/2)+4, int(self.grid_size/2)+6] = 1
        
        self.grid[int(self.grid_size/2)+2, int(self.grid_size/2)-1] = 1
        self.grid[int(self.grid_size/2)+3, int(self.grid_size/2)-1] = 1
        self.grid[int(self.grid_size/2)+4, int(self.grid_size/2)-1] = 1
        self.grid[int(self.grid_size/2)+1, int(self.grid_size/2)-2] = 1
        self.grid[int(self.grid_size/2)+1, int(self.grid_size/2)-3] = 1
        self.grid[int(self.grid_size/2)+1, int(self.grid_size/2)-4] = 1
        self.grid[int(self.grid_size/2)+6, int(self.grid_size/2)-2] = 1
        self.grid[int(self.grid_size/2)+6, int(self.grid_size/2)-3] = 1
        self.grid[int(self.grid_size/2)+6, int(self.grid_size/2)-4] = 1
        self.grid[int(self.grid_size/2)+2, int(self.grid_size/2)-6] = 1
        self.grid[int(self.grid_size/2)+3, int(self.grid_size/2)-6] = 1
        self.grid[int(self.grid_size/2)+4, int(self.grid_size/2)-6] = 1

        self.grid[int(self.grid_size/2)-2, int(self.grid_size/2)-1] = 1
        self.grid[int(self.grid_size/2)-3, int(self.grid_size/2)-1] = 1
        self.grid[int(self.grid_size/2)-4, int(self.grid_size/2)-1] = 1
        self.grid[int(self.grid_size/2)-1, int(self.grid_size/2)-2] = 1
        self.grid[int(self.grid_size/2)-1, int(self.grid_size/2)-3] = 1
        self.grid[int(self.grid_size/2)-1, int(self.grid_size/2)-4] = 1
        self.grid[int(self.grid_size/2)-6, int(self.grid_size/2)-2] = 1
        self.grid[int(self.grid_size/2)-6, int(self.grid_size/2)-3] = 1
        self.grid[int(self.grid_size/2)-6, int(self.grid_size/2)-4] = 1
        self.grid[int(self.grid_size/2)-2, int(self.grid_size/2)-6] = 1
        self.grid[int(self.grid_size/2)-3, int(self.grid_size/2)-6] = 1
        self.grid[int(self.grid_size/2)-4, int(self.grid_size/2)-6] = 1        

        self.grid[int(self.grid_size/2)-2, int(self.grid_size/2)+1] = 1
        self.grid[int(self.grid_size/2)-3, int(self.grid_size/2)+1] = 1
        self.grid[int(self.grid_size/2)-4, int(self.grid_size/2)+1] = 1
        self.grid[int(self.grid_size/2)-1, int(self.grid_size/2)+2] = 1
        self.grid[int(self.grid_size/2)-1, int(self.grid_size/2)+3] = 1
        self.grid[int(self.grid_size/2)-1, int(self.grid_size/2)+4] = 1
        self.grid[int(self.grid_size/2)-6, int(self.grid_size/2)+2] = 1
        self.grid[int(self.grid_size/2)-6, int(self.grid_size/2)+3] = 1
        self.grid[int(self.grid_size/2)-6, int(self.grid_size/2)+4] = 1
        self.grid[int(self.grid_size/2)-2, int(self.grid_size/2)+6] = 1
        self.grid[int(self.grid_size/2)-3, int(self.grid_size/2)+6] = 1
        self.grid[int(self.grid_size/2)-4, int(self.grid_size/2)+6] = 1

    # SEARCH FOR MORE PATTERNS ONLINE -OR CHECK THE SLIDES FROM THE COURSE- AND IMPLEMENT THEM HERE!

#===============================================================================================================
# VISUALISATION OF THE MODEL DYNAMICS USING THE streamlit FRAMEWORK (see more on https://streamlit.io/)

#--------------------------------------------------------------------------------------------------
# Title of the visualisation - shows on screen

st.title("Conway's Game of Life")

#--------------------------------------------------------------------------------------------------
# Methods to interactively collect input variables

N = st.sidebar.slider("Population Size", 100, 1000, 500)
pattern = st.sidebar.radio("Initial Pattern", ('Beacon', 'Block', 'Glider', 'Pulsar'))
boundary = st.sidebar.radio("Boundary Conditions", ('Periodic', 'Finite'))
no_iter = st.sidebar.number_input("Number of Iterations", 10)
speed = st.sidebar.number_input("Speed Simulation", 0.0, 1.0, 0.7)


#--------------------------------------------------------------------------------------------------
# Initialise the object   - Note that when one runs the code, the selected parameters will be passed here during the initialisation of the object via "self" method

gamelife = GameOfLife(N, pattern, boundary)

#--------------------------------------------------------------------------------------------------
# Draw the lattice at the initial time
# Create placeholders for plot, iteration text, and progress bar
plot_placeholder = st.empty()

# This functions shows a progress bar
show_iteration = st.empty()
progress_bar = st.progress(0)

# Display the initial state
# lattice that will show the positions of the random walkers at the current time step
fig, ax = plt.subplots()
ax.axis('off')
cmap = ListedColormap(['white', 'red'])
ax.pcolormesh(gamelife.grid, cmap=cmap, edgecolors='k', linewidths=0.5)

# This function draws the figure (in object) fig on the screen
plot_placeholder.pyplot(fig)

# Close the figure instances
plt.close(fig)

# This functions shows a progress bar
show_iteration.text("Step 0")

#===============================================================================================================
# RUN THE DYNAMICS (IF THE USER CLICKS ON "Run") FOLLOWING THE RULES DEFINED ABOVE IN "RUN"

if st.sidebar.button('Run'):

    # Run the simulation for no_iter iterations, i.e. total time of the simulation
    # Repeat routines below for each time step i
    for i in range(no_iter):

        # Add a little time delay, otherwise the patterns update too fast
        sleep(1.0-speed)

        # Call the method "Run" with the interaction rules
        gamelife.run()

        #--------------------------------------------------------------------
        # Visualisation of the evolution
        # Create a new figure for the updated grid state
        fig, ax = plt.subplots()
        
        ax.axis('off')
        ax.pcolormesh(gamelife.grid, cmap=cmap, edgecolors='k', linewidths=0.5)

        # Draws the figure (in the object plt) on the screen
        plot_placeholder.pyplot(fig)
        
        # Closes all the figures (to replot them in the next time step)
        plt.close("all")

        # Updates the progress bar
        show_iteration.text("Step %d" %(i+1))
        progress_bar.progress( (i+1.0)/no_iter )

        #--------------------------------------------------------------------
