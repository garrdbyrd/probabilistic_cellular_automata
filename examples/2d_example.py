# from PCA.model import model_execute
# from PCA.animator import animate
import PCA.model

seed = 64

parameters = (
    seed,  # Seed
    100,  # X Length
    100,  # Y Length
    1,  # Z Length
    1,  # Initial Infection
    0.1,  # Local Probability
    0.00001,  # Global Probability
    100,  # Run length
    10,  # Run quantity
    False,  # Moore
)

model_execute(parameters)
animate(seed, 0, "png")
