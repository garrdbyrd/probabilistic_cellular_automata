__author__ = "Garrett Dal Byrd"
__version__ = "0.0.1"
__email__ = "gbyrd4@vols.utk.edu"
__status__ = "Development"

from os import mkdir, chdir
from csv import writer
from mesa import Agent, Model  # type: ignore
from mesa.space import NetworkGrid  # type: ignore
from mesa.time import StagedActivation  # type: ignore
from networkx import grid_graph  # type: ignore
from numpy import empty, zeros, asarray
from numpy.random import RandomState, default_rng, binomial
from h5py import File
from math import sqrt
from scipy.spatial.distance import cityblock as L1


class State:
    """States from which cells are assigned. TEMP is used to achieve synchronous activation."""

    SUSCEPTIBLE = 0
    INFECTED = 1
    TEMP = -1


class AutomatonNetwork(Model):
    """A model for a 1,2,3-dimensional probabilistic cellular automaton of rectangular shape."""

    def __init__(self, parameters, iteration, seed=None):
        """Initializes an AutomatonNetwork with the given parameters.

        Parameters:
            parameters  (tuple):    Tuple of parameters given to the model.
            iteration   (int):      For multiples runs, designates which run the model is currently executing.
            seed        (int):      Seed for a given run of this model.

        Attributes:
            x_length        (int):              Length (number of cells) of x axis. Must be greater than 0.
            y_length        (int):              Length (number of cells) of y axis. Must be greater than 0.
            z_length        (int):              Length (number of cells) of z axis. Must be greater than 0.
            population      (int):              Total number of cells in the model. Equal to the product x_length * y_length * z_length.
            init_infection  (int):              Number of cells that are infected at timestep 0. Must be less than or equal to total population.
            local_prob      (float):            Probability that 1 cell in the local neighborhood of another given cell will become infected. Must be >=0 and <=1.
            global_prob     (float):            Probability that 1 cell in the global neighborhood will become infected. Must be >=0 and <=1.
            run_length      (int):              Number of time steps that the model will run.
            moore           (bool):             Shape of local neighborhood. If True, then local neighborhoods are Moore neighborhoods. If False, local neighbhorhoods are von Neumann neighborhoods.

            iteration       (int):              When running a model multiple times with identical parameters, indicates which iteration (run) the model is currently executing.
            time            (int):              Indicates which time step the current run is executing. Begins at 0.

            proportions     (array):            NumPy array which stores the INFECTED/population proportion data.
            # v RENAME 'data'
            data            (array):            NumPy array which stores the state of every cell at every time step.
            # v Change from type list to np.array
            IO_distance     (list):             List which stores the L1 distance from the initially infected cell and the time at which the cell becomes infected.

            G               (graph):            1, 2, 3 dimensional rectangular NetworkX graph which serves to
            grid            (NetworkGrid):      NetworkX NetworkGrid which allows for the placement of cells and retrieval of model information.
            schedule        (StagedActivation): Mesa StagedActivation class which allows for discrete time steps in the model.

            I0              (list):             List of initially infected cells.
        """

        # Assigning parameters
        self.x_length = parameters[1]
        self.y_length = parameters[2]
        self.z_length = parameters[3]
        self.population = self.x_length * self.y_length * self.z_length
        self.init_infection = parameters[4]
        self.local_prob = parameters[5]
        self.global_prob = parameters[6]
        self.run_length = parameters[7]
        self.moore = parameters[9]
        self.iteration = iteration
        self.time = 0

        # Data
        self.proportions = empty((self.run_length, 2))
        self.data = zeros(
            (self.x_length, self.y_length, self.z_length, self.run_length)
        )  # shape of data is: x,y,z,t
        self.I0_distance_data = []

        # Creating grid
        self.G = grid_graph(dim=(self.z_length, self.y_length, self.x_length))
        self.grid = NetworkGrid(self.G)
        self.schedule = StagedActivation(
            self, stage_list=["temp_stage", "infect_stage"]
        )

        # Create cells
        for i, node in enumerate(self.G.nodes()):
            cell = AutomatonCell(i, self, State.SUSCEPTIBLE)
            self.schedule.add(cell)
            self.grid.place_agent(cell, node)

        # Initial infection
        infected_nodes = self.random.sample(self.G.nodes(), self.init_infection)
        self.I0 = infected_nodes[0]
        for cell in self.grid.get_cell_list_contents(infected_nodes):
            cell.state = State.INFECTED
        self.running = True

        # Calculate I0 distance
        for cell in self.grid.get_all_cell_contents():
            cell.I0_distance = L1(cell.pos, self.I0)

    def step(self):
        """Progresses the time step of the model by 1. Also performs appropriate infections and checks if all cells are infected."""
        self.check_global_infection_status()
        self.update_proportions()
        self.update_data()
        self.global_infect()
        self.schedule.step()
        self.time += 1

    def run_model(self):
        """Runs the parent AutomatonNetwork. This should be called once per run."""
        try:
            mkdir(f"run_{self.iteration}")
        except FileExistsError:
            pass
        chdir(f"run_{self.iteration}")
        for _ in range(self.run_length):
            self.step()
            if self.check_global_infection_status():
                break

        # Save proportions
        with File("proportions.hdf5", "w") as f:
            dataset = f.create_dataset("default", data=self.proportions)
        # Save data
        with File("data.hdf5", "w") as f:
            dataset = f.create_dataset("default", data=self.data)
        # Save I0 distance data
        self.I0_distance_array = asarray(self.I0_distance_data)
        with File("I0_distance_data.hdf5", "w") as f:
            dataset = f.create_dataset("default", data=self.I0_distance_array)

        chdir("..")

    def global_infect(self):
        """
        Calculates the number of cells in the global neighborhood which will become infected.

        The number of cells to be infected is calculated as a Bernoulli experiment.
        """
        susceptible_nodes = [
            cell for cell in self.schedule.cells if cell.state is State.SUSCEPTIBLE
        ]
        infected_nodes = [
            cell for cell in self.schedule.cells if cell.state is State.INFECTED
        ]
        try:
            to_be_infected = self.random.sample(
                susceptible_nodes,
                min(
                    binomial(
                        len(susceptible_nodes) * len(infected_nodes),
                        self.global_prob,
                    ),
                    len(susceptible_nodes),
                ),
            )

        except OverflowError:  # Use normal approximation of binomial distribution
            to_be_infected = self.random.sample(
                susceptible_nodes,
                min(
                    max(
                        0,
                        round(
                            self.random.normalvariate(
                                len(susceptible_nodes)
                                * len(infected_nodes)
                                * self.global_prob,
                                sqrt(
                                    len(susceptible_nodes)
                                    * len(infected_nodes)
                                    * self.global_prob
                                    * (1 - self.global_prob)
                                ),
                            )
                        ),
                    ),
                    len(susceptible_nodes),
                ),
            )

        for cell in to_be_infected:
            cell.state = State.TEMP

    def update_proportions(self):
        """Updates the proporations array for the given time step."""
        self.proportions[self.time][0] = self.number_state(0)
        self.proportions[self.time][1] = self.number_state(1)

    def update_data(self):
        """Updates the position/state array for the given time step."""
        infected_nodes = [
            cell for cell in self.schedule.cells if cell.state is State.INFECTED
        ]
        for cell in infected_nodes:
            self.data[cell.pos[0]][cell.pos[1]][cell.pos[2]][self.time] = 1

    def number_state(self, state):
        """Returns the number of cells of arg state on the network."""
        return sum(
            [1 for cell in self.grid.get_all_cell_contents() if cell.state is state]
        )

    def check_global_infection_status(self):
        """Checks if proportion of infected cells is 1 and returns a bool. If True, then the current run is stopped and the remained of the data is automatically filled appropriately."""
        infected_nodes = [
            cell for cell in self.schedule.cells if cell.state is State.INFECTED
        ]
        if len(infected_nodes) == self.population:
            for t in range(self.run_length - self.time):
                self.data[:, :, :, t + self.time] = 1
                self.proportions[t + self.time][0] = self.number_state(0)
                self.proportions[t + self.time][1] = self.number_state(1)
            return True
        else:
            return False


class AutomatonCell(Agent):
    """A single cell for a probabilistic cellular automaton model."""

    def __init__(self, unique_id, model, initial_state):
        """
        Initializes an AutomatonCell object.

        Parameters:
            unique_id       (integers):     A unique integer ID assigned to every cell on the model.
            model           (Model):        A mesa Model object. Used as the framework of this model.
            initial_state   (STATE):        The initial state of the cell. Will only be INFECTED if the cell is infected upon model initialization. Otherwise, will be SUSCEPTIBLE.

        Attributes:
            state           (STATE):    The state of this cell. Will be INFECTED of SUSCEPTIBLE. TEMP is only used to assure simultaneous activation.
            local_prob      (float):    The probability that 1 cell in this cell's local neighborhood will become infected if state is INFECTED.
            moore           (bool):     If True, this cell's local neighborhood is a Moore neighborhood. If False, von Neumann.
            I0_distance     (integer):  This cell's L1 (Manhattan) distance from the initially infected cell.
        """

        super().__init__(unique_id, model)
        self.state = initial_state
        self.local_prob = model.local_prob
        self.moore = model.moore
        self.I0_distance = 0

    def infect_neighbors(self):
        """Infects a number of cells in the local neighborhood of this cell."""
        neighbor_nodes = self.model.grid.get_neighbors(
            self.pos, include_center=False
        )  # RE-implement Moore neighborhood
        susceptible_neighbors = [
            cell
            for cell in self.model.grid.get_cell_list_contents(neighbor_nodes)
            if cell.state is State.SUSCEPTIBLE
        ]
        for cell in susceptible_neighbors:
            if self.random.uniform(0, 1) < self.local_prob:
                cell.state = State.TEMP

    def temp_stage(self):
        """Temporary stage utilizing class State TEMP to achieve synchronous activation of all cells."""
        if self.state is State.INFECTED:
            self.infect_neighbors()

    def infect_stage(self):
        """Stage following temp_stage which converts all cells of state TEMP to state INFECTED."""
        if self.state is State.TEMP:
            self.state = State.INFECTED
            self.model.I0_distance_data.append([self.I0_distance, self.model.time])


def model_execute(parameters):
    """
    Constructs the appropriate files/directories for a model with the given parameters, and also runs that model.

    Parameters:
            parameters  (tuple):    Tuple of parameters given to the model.
    """
    try:
        mkdir("runs")
    except:
        FileExistsError
    chdir("runs")
    mkdir(str(parameters[0]))
    chdir(str(parameters[0]))

    with open("parameters.csv", "w", newline="") as parameter_csv:
        parameter_writer = writer(parameter_csv)
        parameter_writer.writerow(
            [
                "seed",
                "x_length",
                "y_length",
                "z_length",
                "init_infection",
                "local_prob",
                "global_prob",
                "run_length",
                "run_quantity",
                "moore",
            ]
        )
        parameter_writer.writerow([parameter for parameter in parameters])

    for iteration in range(parameters[8]):
        run = AutomatonNetwork(
            parameters, iteration, seed=int(str(f"{parameters[0]}{iteration}"))
        )
        run.run_model()
    chdir("..\..")
