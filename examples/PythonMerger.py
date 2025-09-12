from PyStellarMerger.StellarMerger import StellarMerger
import json

# Load PyMMAMS input parameters from JSON file
with open("pymmams_input.json", "r") as f:
    pymmams_parameters = json.load(f)

# Load EntropySorting input parameters from JSON file
with open("entropysorting_input.json", "r") as f:
    entropysorting_parameters = json.load(f)

# Perform the merger using the PyMMAMS method
pymmams_obj = StellarMerger(pymmams_parameters)
pymmams_obj.PyMMAMS()

# Perform the merger using the EntropySorting method
# entropysorting_obj = StellarMerger(entropysorting_parameters)
# entropysorting_obj.EntropySorting()
