from utils import *
# 1. Setup Model and Parameters
model = pybamm.lithium_ion.DFN()
params = pybamm.ParameterValues("Chen2020")

rate = 4
total_time = 0.5*abs(3600/rate)
time_steps = int(0.5*100)
# Pack configuration
num_cells_parallel = 4
# configuration = 'U'

r_busbar = 20*2.5e-5  # Busbar resistance between cells

sols, history = run_pack(
    model=model,
    params=params,
    rate=rate,
    total_time=total_time,
    time_steps=time_steps,
    num_cells_parallel=num_cells_parallel,
    r_busbar=r_busbar,
    initial_soc=1,
)

pybamm.dynamic_plot(
    sols, 
    output_variables=["Voltage [V]", "Current [A]", "Positive particle surface concentration"],
)

print("Simulation complete.")

def plot_results(history, num_cells):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    time = history["time"]
    colors = plt.cm.viridis(np.linspace(0, 1, num_cells))

    # Plot Currents
    for i in range(num_cells):
        axes[0].plot(time, history["currents"][i], label=f'Cell {i+1}', color=colors[i])
    axes[0].set_title("Cell Currents")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Current [A]")
    axes[0].grid(True, alpha=0.5)

    # Plot Voltages
    for i in range(num_cells):
        axes[1].plot(time, history["voltages"][i], label=f'Cell {i+1}', color=colors[i])
    axes[1].set_title("Terminal Voltages")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Voltage [V]")
    axes[1].grid(True, alpha=0.5)

    # Plot Resistances
    for i in range(num_cells):
        axes[2].plot(time, history["resistances"][i], label=f'Cell {i+1}', color=colors[i])
    axes[2].set_title("Internal Resistance (Linearized)")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Resistance [Ohm]")
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()

plot_results(history, num_cells_parallel)
