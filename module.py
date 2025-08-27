import numpy as np
from mpmath import invertlaplace, mp
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import multiprocessing
from functools import partial
import time
import sympy as sp
import os
import warnings
import ipywidgets as widgets
from IPython.display import display

warnings.filterwarnings("ignore")
mp.dps = 12

_voltage_funcs_global = None

def _initialize_worker_globals(voltage_funcs):
    global _voltage_funcs_global
    _voltage_funcs_global = voltage_funcs

def create_symbolic_impedance_matrix(num_loops, edges, loops, s):
    max_node = max(max(node_list) for node_list in loops)
    loop_matrix_np = np.zeros((num_loops, max_node), dtype=int)
    for i, nodes in enumerate(loops):
        for node in nodes:
            if node - 1 < max_node:
                loop_matrix_np[i, node - 1] = 1
    Z = sp.zeros(num_loops, num_loops)
    for i in range(num_loops):
        total_impedance = 0
        for n1, n2, r_elem in edges:
            if (n1 - 1 < max_node and loop_matrix_np[i, n1 - 1] == 1) and \
               (n2 - 1 < max_node and loop_matrix_np[i, n2 - 1] == 1):
                total_impedance += r_elem
        Z[i, i] = total_impedance
    for n1, n2, r_elem in edges:
        loops_with_edge = []
        for i in range(num_loops):
            if (n1 - 1 < max_node and loop_matrix_np[i, n1 - 1] == 1) and \
               (n2 - 1 < max_node and loop_matrix_np[i, n2 - 1] == 1):
                loops_with_edge.append(i)
        if len(loops_with_edge) == 2:
            i, j = loops_with_edge[0], loops_with_edge[1]
            Z[i, j] = Z[j, i] = -r_elem
    return Z, loop_matrix_np

def get_symbolic_branch_currents(edges, loop_matrix_np, I_symbolic):
    branch_currents = {}
    n_loops = loop_matrix_np.shape[0]
    max_node_idx = loop_matrix_np.shape[1] - 1
    for n1, n2, _ in edges:
        edge_loops_indices = []
        for i in range(n_loops):
            if (n1 - 1 <= max_node_idx and loop_matrix_np[i, n1 - 1] == 1) and \
               (n2 - 1 <= max_node_idx and loop_matrix_np[i, n2 - 1] == 1):
                edge_loops_indices.append(i)
        current = 0
        if len(edge_loops_indices) == 1:
            current = I_symbolic[edge_loops_indices[0]]
        elif len(edge_loops_indices) == 2:
            current = I_symbolic[edge_loops_indices[1]] - I_symbolic[edge_loops_indices[0]]
        branch_currents[(n1, n2)] = current
    return branch_currents

def create_precompiled_voltage_functions(Z, V, edges, loop, capacitor_branches_map, r, s):
    print("Pre-calculating symbolic voltage functions...")
    start_symbolic_time = time.time()
    Z_inv = Z.inv()
    I_symbolic = Z_inv * V
    symbolic_branch_currents = get_symbolic_branch_currents(edges, loop, I_symbolic)
    voltage_functions = {}
    symbolic_voltage_functions = {}
    for vkey, branch_nodes in capacitor_branches_map.items():
        v_symbolic = symbolic_branch_currents.get(branch_nodes, 0) * r
        voltage_functions[vkey] = sp.lambdify(s, v_symbolic, 'numpy')
        symbolic_voltage_functions[vkey] = v_symbolic
    print(f"Symbolic pre-calculation finished in {time.time() - start_symbolic_time:.2f} s")
    return voltage_functions, symbolic_voltage_functions

def calculate_voltage_at_t(t_val, voltage_key):
    if t_val == 0:
        return 0.0
    try:
        s_func_internal = _voltage_funcs_global[voltage_key]
        def s_func_for_laplace(s_laplace):
            return mp.mpc(s_func_internal(complex(s_laplace)))
        val = invertlaplace(s_func_for_laplace, t_val, method='talbot', degree=8)
        return float(val.real)
    except Exception:
        return 0.0

def compute_voltages(t_vals, voltage_functions, voltage_map, decimal_places, num_workers=None):
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 4)
    voltage_data, max_voltages, max_times = {}, {}, {}
    chunk_size = max(1, len(t_vals) // (num_workers * 5))
    with multiprocessing.Pool(processes=num_workers, initializer=_initialize_worker_globals, initargs=(voltage_functions,)) as pool:
        for vkey, vinfo in voltage_map.items():
            print(f"\nCalculating {vkey} ({vinfo['capacitor']}) with {num_workers} workers...")
            worker_func = partial(calculate_voltage_at_t, voltage_key=vkey)
            v_results = pool.map(worker_func, t_vals, chunksize=chunk_size)
            v_results = np.array(v_results)
            voltage_data[vkey] = v_results
            max_idx = np.argmax(np.abs(v_results))
            max_voltages[vkey] = v_results[max_idx]
            max_times[vkey] = t_vals[max_idx]
            print(f"Max Voltage ({vinfo['capacitor']}): {max_voltages[vkey]:.{decimal_places}f} V at time {max_times[vkey]:.{decimal_places}f} s")
    return voltage_data, max_voltages, max_times

def generate_time_points(t_start, t_end, num_points=500):
    log_start = np.log10(max(t_start, 1e-6))
    log_end = np.log10(t_end)
    log_points = np.linspace(log_start, log_end, num_points)
    linear_points_initial = np.linspace(0, 0.1 * t_end, num=int(num_points * 0.1))
    all_points = np.concatenate([linear_points_initial, 10**log_points])
    return np.unique(np.sort(all_points))

# -----------------------------
# سایر توابع generate_plots / generate_table / generate_max_voltage_table همان قبلی هستند
# -----------------------------

def generate_outputs(t_vals, voltage_data, max_voltages, max_times, voltage_map,
                     plot_filename="plots.pdf", table_filename="tables.pdf",
                     max_voltage_table_filename="max_voltages_table.pdf", decimal_places=4):
    try:
        generate_plots(t_vals, voltage_data, max_voltages, max_times, voltage_map, plot_filename, decimal_places)
        generate_table(t_vals, voltage_data, voltage_map, table_filename, decimal_places)
        generate_max_voltage_table(max_voltages, max_times, voltage_map, max_voltage_table_filename, decimal_places)
    except (IOError, PermissionError) as e:
        print(f"خطا در نوشتن فایل‌های خروجی: {e}")
        raise

def display_interactive_widgets(t_vals, voltage_data, voltage_map, decimal_places):
    capacitor_options = [(vinfo['capacitor'], vkey) for vkey, vinfo in voltage_map.items()]
    capacitor_dropdown = widgets.Dropdown(options=capacitor_options, description='Capacitor:', value='v1')
    mode_selector = widgets.RadioButtons(options=['Voltage to Time', 'Time to Voltage'], description='Mode:', disabled=False)
    max_voltage_all = max(np.max(np.abs(v)) for v in voltage_data.values()) if voltage_data else 0
    volt_step = 0.1
    v_levels = np.arange(volt_step, max_voltage_all + volt_step, volt_step)
    voltage_options = [(f"{v:.{decimal_places}f} V", v) for v in v_levels]
    voltage_dropdown = widgets.Dropdown(options=voltage_options, description='Voltage:', value=v_levels[0] if len(v_levels) > 0 else 0)
    time_input = widgets.FloatText(value=10.0, description='Time (s):', step=0.1)
    calculate_button = widgets.Button(description='Calculate', button_style='success')

    # تغییر: خروجی باکس HTML
    output_label = widgets.HTML(
        value="",
        layout=widgets.Layout(border='solid 2px black', padding='10px', margin='5px', width='500px')
    )

    voltage_box = widgets.HBox([voltage_dropdown])
    time_box = widgets.HBox([time_input])

    def on_mode_change(change):
        if change['new'] == 'Voltage to Time':
            voltage_box.layout.display = 'flex'
            time_box.layout.display = 'none'
        else:
            voltage_box.layout.display = 'none'
            time_box.layout.display = 'flex'
        output_label.value = ''

    mode_selector.observe(on_mode_change, names='value')

    def on_button_clicked(b):
        capacitor_key = capacitor_dropdown.value
        capacitor_name = voltage_map[capacitor_key]['capacitor']
        if mode_selector.value == 'Voltage to Time':
            voltage_target = voltage_dropdown.value
            v_abs = np.abs(voltage_data[capacitor_key])
            indices = np.where(v_abs >= voltage_target)[0]
            if len(indices) > 0:
                result_time = t_vals[indices[0]]
                output_label.value = f"<div style='font-size:18px; font-weight:bold; color:blue;'>Time for {capacitor_name} to reach {voltage_target:.{decimal_places}f} V: {result_time:.{decimal_places}f} s</div>"
            else:
                output_label.value = f"<div style='font-size:18px; font-weight:bold; color:red;'>{capacitor_name} never reaches {voltage_target:.{decimal_places}f} V</div>"
        else:
            time_target = time_input.value
            if not t_vals.any() or time_target < t_vals[0] or time_target > t_vals[-1]:
                output_label.value = f"<div style='font-size:18px; font-weight:bold; color:red;'>Time must be between {t_vals[0]:.{decimal_places}f}s and {t_vals[-1]:.{decimal_places}f}s</div>"
                return
            idx = np.argmin(np.abs(t_vals - time_target))
            result_voltage = voltage_data[capacitor_key][idx]
            output_label.value = f"<div style='font-size:18px; font-weight:bold; color:green;'>Voltage of {capacitor_name} at {t_vals[idx]:.{decimal_places}f} s: {result_voltage:.{decimal_places}f} V</div>"

    calculate_button.on_click(on_button_clicked)
    on_mode_change({'new': mode_selector.value})
    print("\nSelect mode, capacitor, and threshold, then click Calculate:")
    display(widgets.VBox([mode_selector, capacitor_dropdown, voltage_box, time_box, calculate_button, output_label]))

# -----------------------------
# تابع نهایی برای گزارش‌دهی
# -----------------------------
def run_final_reports(t_vals, voltage_data, max_voltages, max_times, voltage_map,
                      plot_filename="plots.pdf", table_filename="tables.pdf",
                      max_voltage_table_filename="max_table.pdf",
                      decimal_places=4, start_time_total=None):
    # تولید خروجی‌ها
    generate_outputs(
        t_vals, voltage_data, max_voltages, max_times, voltage_map,
        plot_filename=plot_filename,
        table_filename=table_filename,
        max_voltage_table_filename=max_voltage_table_filename,
        decimal_places=decimal_places
    )

    # ویجت‌های تعاملی
    display_interactive_widgets(t_vals, voltage_data, voltage_map, decimal_places)

    # گزارش نهایی
    print(f"\nReports generated:")
    if os.path.exists(plot_filename):
        print(f"Plots: {plot_filename} | Size: {os.path.getsize(plot_filename)/1024:.2f} KB")
    if os.path.exists(table_filename):
        print(f"Table (Threshold): {table_filename} | Size: {os.path.getsize(table_filename)/1024:.2f} KB")
    if os.path.exists(max_voltage_table_filename):
        print(f"Table (Max Voltages): {max_voltage_table_filename} | Size: {os.path.getsize(max_voltage_table_filename)/1024:.2f} KB")
    if start_time_total:
        print(f"Total Computation Time: {time.time() - start_time_total:.2f} s")
