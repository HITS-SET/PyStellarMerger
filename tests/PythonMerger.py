from PyStellarMerger.StellarMerger import StellarMerger
import json

with open("input.json", "r") as f:
    parameters = json.load(f)

mrgr_obj = StellarMerger(parameters)

mrgr_obj.PyMMAMS()