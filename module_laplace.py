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

warnings.filterwarnings("ignore")
mp.dps = 12  # دقت محاسبات عددی

# متغیر گلوبال برای نگهداری توابع ولتاژ بهینه‌شده در فرآیندهای موازی
_voltage_funcs_global = None

def _initialize_worker_globals(voltage_funcs):
    """مقداردهی اولیه متغیرهای گلوبال برای فرآیندهای کارگر."""
    global _voltage_funcs_global
    _voltage_funcs_global = voltage_funcs

def create_symbolic_impedance_matrix(n, edges, loops, s):
    """ساخت ماتریس امپدانس سیمبولیک (Z) برای مدار."""
    max_node = max(max(node_list) for node_list in loops)
    loop_matrix_np = np.zeros((n, max_node), dtype=int)
    for i, nodes in enumerate(loops):
        for node in nodes:
            if node - 1 < max_node:
                loop_matrix_np[i, node - 1] = 1
    Z = sp.zeros(n, n)
    for i in range(n):
        total_impedance = 0
        for n1, n2, r_elem in edges:
            if (n1 - 1 < max_node and loop_matrix_np[i, n1 - 1] == 1) and \
               (n2 - 1 < max_node and loop_matrix_np[i, n2 - 1] == 1):
                total_impedance += r_elem
        Z[i, i] = total_impedance
    for n1, n2, r_elem in edges:
        loops_with_edge = []
        for i in range(n):
            if (n1 - 1 < max_node and loop_matrix_np[i, n1 - 1] == 1) and \
               (n2 - 1 < max_node and loop_matrix_np[i, n2 - 1] == 1):
                loops_with_edge.append(i)
        if len(loops_with_edge) == 2:
            i, j = loops_with_edge[0], loops_with_edge[1]
            Z[i, j] = Z[j, i] = -r_elem
    return Z, loop_matrix_np

def get_symbolic_branch_currents(edges, loop_matrix_np, I_symbolic):
    """محاسبه جریان‌های شاخه به صورت سیمبولیک از جریان‌های حلقه سیمبولیک."""
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
    """
    معادلات را به صورت سیمبولیک حل کرده و توابع ولتاژ بهینه‌شده (lambdified) را برمی‌گرداند.
    """
    print("Pre-calculating symbolic voltage functions (this is a one-time cost)...")  # اصلاح خطای سینتکسی
    start_symbolic_time = time.time()
    Z_inv = Z.inv()
    I_symbolic = Z_inv * V
    symbolic_branch_currents = get_symbolic_branch_currents(edges, loop, I_symbolic)
    voltage_functions = {}
    for vkey, branch_nodes in capacitor_branches_map.items():
        v_symbolic = symbolic_branch_currents.get(branch_nodes, 0) * r
        voltage_functions[vkey] = sp.lambdify(s, v_symbolic, 'numpy')
    print(f"Symbolic pre-calculation finished in {time.time() - start_symbolic_time:.2f} s")
    return voltage_functions

def calculate_voltage_at_t(t_val, voltage_key):
    """انجام معکوس لاپلاس عددی با استفاده از تابع از پیش کامپایل شده."""
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

def compute_voltages(t_vals, voltage_functions, voltage_map, num_workers=None):
    """محاسبه ولتاژها با استفاده از multiprocessing و توابع از پیش کامپایل شده."""
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
            print(f"Max Voltage ({vinfo['capacitor']}): {max_voltages[vkey]:.4f} V at time {max_times[vkey]:.2f} s")
    return voltage_data, max_voltages, max_times

def generate_time_points(t_start, t_end, num_points=500):
    """تولید نقاط زمانی به صورت بهینه."""
    log_start = np.log10(max(t_start, 1e-6))
    log_end = np.log10(t_end)
    log_points = np.linspace(log_start, log_end, num_points)
    linear_points_initial = np.linspace(0, 0.1 * t_end, num=int(num_points * 0.1))
    all_points = np.concatenate([linear_points_initial, 10**log_points])
    return np.unique(np.sort(all_points))

def generate_plots(t_vals, voltage_data, max_voltages, max_times, voltage_map, plot_filename):
    """تولید فایل PDF شامل نمودارها با زمینه شطرنجی خوانا."""
    with PdfPages(plot_filename) as pdf:
        plt.rcParams.update({'font.size': 8, 'axes.titlesize': 10, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7})
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes = axes.flatten()
        colors = ['b', 'g', 'r', 'c']
        for idx, (vkey, vinfo) in enumerate(voltage_map.items()):
            ax = axes[idx]
            ax.plot(t_vals, voltage_data[vkey], colors[idx], linewidth=1.5, label=vinfo['capacitor'])
            ax.plot(max_times[vkey], max_voltages[vkey], 'ro', markersize=5, label=f'Max: {max_voltages[vkey]:.4f} V at {max_times[vkey]:.2f} s')
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Voltage (V)'); ax.set_title(f'{vinfo["label"]} - {vinfo["capacitor"]}')
            ax.grid(True, which='major', linestyle='-', linewidth='0.7', color='black', alpha=0.6)
            ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)
            ax.minorticks_on(); ax.legend(fontsize=7)
        plt.tight_layout(pad=1.5); pdf.savefig(fig, bbox_inches='tight', dpi=300); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, (vkey, vinfo) in enumerate(voltage_map.items()):
            ax.plot(t_vals, voltage_data[vkey], color=colors[idx % len(colors)], linestyle='-', linewidth=1.2, label=f'{vinfo["capacitor"]} (Max: {max_voltages[vkey]:.2f} V)')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Voltage (V)'); ax.set_title('Capacitor Voltages Comparison')
        ax.grid(True, which='major', linestyle='-', linewidth='0.7', color='black', alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)
        ax.minorticks_on(); ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout(pad=1.5); pdf.savefig(fig, bbox_inches='tight', dpi=300); plt.close(fig)

def generate_table(t_vals, voltage_data, voltage_map, table_filename):
    """تولید فایل PDF شامل جدول دوصفحه‌ای با اندازه بهینه و خوانا برای چاپ."""
    max_voltage_all = max(np.max(np.abs(v)) for v in voltage_data.values())
    volt_step = 0.1
    v_levels = np.arange(volt_step, max_voltage_all + volt_step, volt_step)
    table_data = []
    for v_target in v_levels:
        row = [f"{v_target:.2f}"]
        for vkey in voltage_map.keys():
            indices = np.where(np.abs(voltage_data[vkey]) >= v_target)[0]
            row.append(f"{t_vals[indices[0]]:.2f}" if len(indices) > 0 else "-")
        table_data.append(row)
    headers = ["Voltage (V)"] + ["Time " + vinfo['capacitor'] for vinfo in voltage_map.values()]
    print("\nVoltage Threshold Crossing Times:\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # تقسیم جدول به دو صفحه
    total_rows = len(table_data)
    rows_per_page = (total_rows + 1) // 2  # نصف ردیف‌ها در هر صفحه
    num_cols = len(headers)
    col_widths = [0.18] + [0.82 / (num_cols - 1)] * (num_cols - 1)  # ستون ولتاژ کمی باریک‌تر
    
    with PdfPages(table_filename) as pdf:
        for page in range(0, total_rows, rows_per_page):
            fig, ax = plt.subplots(figsize=(8.27, 11.69))  # اندازه A4
            ax.axis('off')
            # انتخاب ردیف‌های صفحه فعلی
            page_data = table_data[page:page + rows_per_page]
            # افزودن سربرگ فقط در صفحه اول
            table = ax.table(cellText=page_data, colLabels=headers if page == 0 else None, 
                            loc='center', cellLoc='center', colWidths=col_widths)
            table.auto_set_font_size(False)
            table.set_fontsize(8)  # فونت کوچک‌تر برای جا شدن
            table.scale(1.0, 0.9)  # مقیاس فشرده‌تر
            fig.subplots_adjust(left=0.15, right=0.85, top=0.92, bottom=0.08)  # حاشیه‌های متعادل
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close(fig)

def generate_outputs(t_vals, voltage_data, max_voltages, max_times, voltage_map, plot_filename="plots.pdf", table_filename="table.pdf"):
    """تولید تمام خروجی‌های PDF (نمودار و جدول)."""
    try:
        generate_plots(t_vals, voltage_data, max_voltages, max_times, voltage_map, plot_filename)
        generate_table(t_vals, voltage_data, voltage_map, table_filename)
    except (IOError, PermissionError) as e:
        print(f"خطا در نوشتن فایل‌های خروجی: {e}")
        raise