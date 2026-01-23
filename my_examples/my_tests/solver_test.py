import liionpack as lp
import pybamm
import numpy as np
import matplotlib.pyplot as plt
from my_examples.run_cell import base_model, advanced_model, discret_points, param_base, param_adv 

# print("pybamm version: ", pybamm.__version__) # must be 24.9

def simul_ida(parameter_values=None):
    model = base_model.new_copy()
    # Add events to the model
    model = lp.add_events_to_model(model)
    # Set up simulation
    sim = pybamm.Simulation(
        model=model,
        parameter_values=parameter_values,
        solver = pybamm.IDAKLUSolver(atol=1e-4, rtol=1e-4),
        var_pts=discret_points,
    )
    return sim

def simul_casadi(parameter_values=None):
    model = base_model.new_copy()
    # Add events to the model
    model = lp.add_events_to_model(model)
    # Set up simulation
    sim = pybamm.Simulation(
        model=model,
        parameter_values=parameter_values,
        solver = pybamm.CasadiSolver(mode="safe"),
        var_pts=discret_points,
    )
    return sim

I_mag = 160.0
OCV_init = 3.5  # used for initial guess
Ri_init = 6e-4  # used for initial guess
R_busbar = 10e-5
R_connection = 1e-4
Np = 4
Ns = 1
Nbatt = Np * Ns
netlist = lp.setup_circuit(
    Np=Np, 
    Ns=Ns, 
    Rb=R_busbar, 
    Rc=R_connection, 
    Ri=Ri_init, 
    V=OCV_init, 
    I=I_mag
)

rate = 1.0
current = rate *  param_base["Nominal cell capacity [A.h]"] * Nbatt # 1C in A
initial_soc = 1.0
final_soc = 0.5
time = 60*(initial_soc - final_soc) / rate  # in minutes

experiment = pybamm.Experiment(
    [
        f"Discharge at {current} A for {time} minutes",
        "Rest for 20 minutes",
    ],
    period= f"{time/100} minutes",
)

output_variables = [
    # "X-averaged negative particle surface concentration",
    # "X-averaged positive particle surface concentration",
    # "X-averaged negative electrode extent of lithiation",
    "X-averaged positive electrode extent of lithiation"
]


print("Starting IDA simulation...", flush=True)

output_ida = lp.solve(
    sim_func = simul_ida,
    netlist=netlist.copy(),
    parameter_values=param_base.copy(),
    experiment=experiment,
    output_variables=output_variables,
    initial_soc=initial_soc,
)
print("IDA simulation finished", flush=True)

# Define period for max step size control as requested


print("Starting CasADI simulation...", flush=True)
output_casa = lp.solve(
    sim_func = simul_casadi,
    netlist=netlist.copy(),
    parameter_values=param_base.copy(),
    experiment=experiment,
    output_variables=output_variables,
    initial_soc=initial_soc,
)
print("CasADI simulation finished", flush=True)


outputs = [output_ida, output_casa]
# outputs = [output_ida]
styles = ["--", "-"]
labels = ["IDA model", "Casadi model"]

# Convenient function to compute SOC from extent of lithiation
def get_soc(output):
    sto_at_100 = 0.005
    sto_at_0 = 0.813
    ext_of_lith = output["X-averaged positive electrode extent of lithiation"]
    return 100*(1 - (ext_of_lith - sto_at_100) / (sto_at_0 - sto_at_100))

fig, axes = plt.subplots(2,1)

ax = axes[0]
for out, style, lab in zip(outputs, styles, labels):
    time = out["Time [s]"]
    soc = get_soc(out)
    avg_soc = np.mean(soc[-1, :])
    print(f"{lab} Average SOC at end of discharge: {avg_soc}", flush=True)
    for cell_number in range(Nbatt):
        soc_cell = soc[:, cell_number]
        ax.plot(time/60, soc_cell, style, 
                label=f"{lab} - Cell {cell_number+1}", 
                color=f"C{cell_number}")

ax.set_xlabel("Time (min)", fontsize=14)
ax.set_ylabel("State of Charge (%)", fontsize=14)

# ax.legend(frameon=False)

ax = axes[1]
for out, style, lab in zip(outputs, styles, labels):
    time = out["Time [s]"]
    print(f"{lab} Total Time: {time[-1]/60:.2f} min", flush=True)
    voltage = out["Terminal voltage [V]"]
    print(f"{lab} Final voltage: {voltage[-1, :]}", flush=True)
    for cell_number in range(Nbatt):
        vol_cell = voltage[:, cell_number]
        ax.plot(time/60, vol_cell, style, 
                label=f"{lab} - Cell {cell_number+1}", 
                color=f"C{cell_number}")
ax.set_xlabel("Time (min)", fontsize=14)
ax.set_ylabel("Terminal Voltage (V)", fontsize=14)
ax.legend(frameon=False)

# plt.savefig("solver_comparison.png")
plt.show()

# lp.plot_cells(output_adv, color="light")
# lp.plot_cells(output_base, color="dark")


plt.show()