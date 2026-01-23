
import liionpack as lp
import pybamm
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
param = pybamm.ParameterValues("Chen2020")

# Create two parameter sets with different capacities
param1 = param.copy()
param2 = param.copy()

cap_key = "Negative electrode active material volume fraction"  # Example key; replace with actual key if different
nominal_cap = param[cap_key]
param2[cap_key] = nominal_cap * 0.1

cap_key2 = "Positive electrode active material volume fraction"  # Example key; replace with actual key if different
nominal_cap = param[cap_key2]
param2[cap_key2] = nominal_cap * 0.1

# Setup circuit
Np = 2
Ns = 1
netlist = lp.setup_circuit(Np=Np, Ns=Ns, Rb=1e-4, Rc=1e-4, Ri=5e-2, V=4.0, I=10.0)

# Experiment
experiment = pybamm.Experiment(
    [
        "Discharge at 30 A for 10 minutes",
    ],
    period="10 seconds",
)

# Solve with list of parameters
print("Starting simulation with variable parameters...", flush=True)
output = lp.solve(
    netlist=netlist,
    parameter_values=[param1, param2],
    experiment=experiment,
    initial_soc=1.0,
    output_variables=["Terminal voltage [V]",
                      "X-averaged positive electrode extent of lithiation"]
)
print("Simulation finished.", flush=True)

# Plot results
time = output["Time [s]"]
voltage = output["Terminal voltage [V]"]

ext_of_lith = output["X-averaged positive electrode extent of lithiation"]


plt.figure()
for i in range(Np):
    # plt.plot(time, soc, label=f"Cell {i+1}")
    plt.plot(ext_of_lith[:, i], voltage[:, i], label=f"Cell {i+1}")

# plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.legend()
plt.title("Voltage divergence due to capacity difference")
plt.show()
