import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Frame, Canvas, Scrollbar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
from algorithms import (
    hill_climbing, simulated_annealing, genetic_algorithm, 
    tabu_search, particle_swarm_optimization, differential_evolution,
    exact_optimization
)
import time


class ScrollableFrame(ttk.Frame):
    """A scrollable frame class that wraps its contents in a scrollable canvas."""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        self.canvas = Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.bind_mousewheel()
        
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
    def bind_mousewheel(self):
        """Bind mousewheel events for scrolling."""
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            
        def _on_mousewheel_linux(event):
            if event.num == 4: 
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:  
                self.canvas.yview_scroll(1, "units")
                
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel) 
        self.canvas.bind_all("<Button-4>", _on_mousewheel_linux)  
        self.canvas.bind_all("<Button-5>", _on_mousewheel_linux)  
    
    def on_canvas_resize(self, event):
        """Update the width of the frame when canvas is resized."""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

class PortfolioOptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Portfolio Optimization Tool")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
     
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        self.current_data = {}  
        self.selected_file = tk.StringVar(value="No file selected")
        self.optimization_results = None  
        
        self.create_widgets()
        
    def create_widgets(self):
        self.main_container = ttk.Frame(self.root)
        self.main_container.grid(row=0, column=0, sticky="nsew")
        self.main_container.columnconfigure(0, weight=1)
        self.main_container.rowconfigure(0, weight=1)
        
        self.scrollable_frame = ScrollableFrame(self.main_container)
        self.scrollable_frame.grid(row=0, column=0, sticky="nsew")
    
        main_frame = ttk.Frame(self.scrollable_frame.scrollable_frame, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        main_frame.columnconfigure(0, weight=1)
        
        title_label = ttk.Label(
            main_frame, 
            text="Portfolio Optimization Tool", 
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=10)
        
        file_frame = ttk.LabelFrame(main_frame, text="Data Selection", padding=10)
        file_frame.pack(fill=tk.X, pady=10)
        
        file_list_frame = ttk.Frame(file_frame)
        file_list_frame.pack(fill=tk.X, pady=5)
        
        file_list_frame.columnconfigure(0, weight=1)
        
        self.file_listbox = tk.Listbox(file_list_frame, height=5, selectmode="extended")
        self.file_listbox.grid(row=0, column=0, sticky="ew")
        
        scrollbar = ttk.Scrollbar(file_list_frame, orient="vertical", command=self.file_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.file_listbox.config(yscrollcommand=scrollbar.set)
        
        file_buttons_frame = ttk.Frame(file_frame)
        file_buttons_frame.pack(fill=tk.X, pady=5)
        
        file_buttons_frame.columnconfigure((0, 1, 2, 3, 4), weight=1)
        
        browse_button = ttk.Button(
            file_buttons_frame, 
            text="Add CSV File", 
            command=self.browse_files
        )
        browse_button.grid(row=0, column=0, padx=2, sticky="ew")
        
        browse_folder_button = ttk.Button(
            file_buttons_frame, 
            text="Add Folder of CSV Files", 
            command=self.browse_folder
        )
        browse_folder_button.grid(row=0, column=1, padx=2, sticky="ew")
        
        remove_button = ttk.Button(
            file_buttons_frame, 
            text="Remove Selected", 
            command=self.remove_selected_file
        )
        remove_button.grid(row=0, column=2, padx=2, sticky="ew")
        
        remove_all_button = ttk.Button(
            file_buttons_frame, 
            text="Remove All", 
            command=self.remove_all_files
        )
        remove_all_button.grid(row=0, column=3, padx=2, sticky="ew")
        
        show_graph_button = ttk.Button(
            file_buttons_frame,
            text="Show Price Data",
            command=self.show_selected_graph
        )
        show_graph_button.grid(row=0, column=4, padx=2, sticky="ew")
        
        constraints_frame = ttk.LabelFrame(main_frame, text="Portfolio Constraints", padding=10)
        constraints_frame.pack(fill=tk.X, pady=10)
        
        constraints_grid = ttk.Frame(constraints_frame)
        constraints_grid.pack(fill=tk.X, expand=True)
        constraints_grid.columnconfigure((0, 1, 2, 3), weight=1)

        ttk.Label(constraints_grid, text="Initial Investment ($):").grid(row=0, column=0, sticky=tk.E, pady=5, padx=5)
        self.initial_investment = ttk.Entry(constraints_grid, width=10)
        self.initial_investment.insert(0, "10000")
        self.initial_investment.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(constraints_grid, text="Risk Aversion (0-1):").grid(row=1, column=0, sticky=tk.E, pady=5, padx=5)
        self.risk_aversion = ttk.Entry(constraints_grid, width=10)
        self.risk_aversion.insert(0, "0.5")
        self.risk_aversion.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(constraints_grid, text="Max Asset Weight (%):").grid(row=2, column=0, sticky=tk.E, pady=5, padx=5)
        self.max_weight = ttk.Entry(constraints_grid, width=10)
        self.max_weight.insert(0, "30")
        self.max_weight.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(constraints_grid, text="Min Return (%):").grid(row=0, column=2, sticky=tk.E, pady=5, padx=5)
        self.min_return = ttk.Entry(constraints_grid, width=10)
        self.min_return.insert(0, "0")
        self.min_return.grid(row=0, column=3, sticky=tk.W, pady=5)
        
        ttk.Label(constraints_grid, text="Max Risk (%):").grid(row=1, column=2, sticky=tk.E, pady=5, padx=5)
        self.max_risk = ttk.Entry(constraints_grid, width=10)
        self.max_risk.insert(0, "30")
        self.max_risk.grid(row=1, column=3, sticky=tk.W, pady=5)
        
        self.allow_short = tk.BooleanVar(value=False)
        short_checkbox = ttk.Checkbutton(
            constraints_grid, 
            text="Allow Short Selling", 
            variable=self.allow_short
        )
        short_checkbox.grid(row=2, column=2, columnspan=2, sticky=tk.W, pady=5)
      
        param_frame = ttk.LabelFrame(main_frame, text="Algorithm Parameters", padding=10)
        param_frame.pack(fill=tk.X, pady=10)
        
        self.param_notebook = ttk.Notebook(param_frame)
        self.param_notebook.pack(fill=tk.BOTH, expand=True)

        meta_frame = ttk.Frame(self.param_notebook)
        self.param_notebook.add(meta_frame, text="Metaheuristics")
        
        meta_notebook = ttk.Notebook(meta_frame)
        meta_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Hill Climbing Tab
        hc_frame = ttk.Frame(meta_notebook, padding=10)
        meta_notebook.add(hc_frame, text="Hill Climbing")
        hc_frame.columnconfigure((0, 1, 2, 3), weight=1)
        
        ttk.Label(hc_frame, text="Iterations:").grid(row=0, column=0, sticky=tk.E, pady=5, padx=5)
        self.hc_iterations = ttk.Entry(hc_frame, width=10)
        self.hc_iterations.insert(0, "1000")
        self.hc_iterations.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(hc_frame, text="Step Size:").grid(row=1, column=0, sticky=tk.E, pady=5, padx=5)
        self.hc_step = ttk.Entry(hc_frame, width=10)
        self.hc_step.insert(0, "0.05")
        self.hc_step.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(hc_frame, text="Restarts:").grid(row=2, column=0, sticky=tk.E, pady=5, padx=5)
        self.hc_restarts = ttk.Entry(hc_frame, width=10)
        self.hc_restarts.insert(0, "5")
        self.hc_restarts.grid(row=2, column=1, sticky=tk.W, pady=5)

        # Simulated Annealing Tab
        sa_frame = ttk.Frame(meta_notebook, padding=10)
        meta_notebook.add(sa_frame, text="Simulated Annealing")
        sa_frame.columnconfigure((0, 1, 2, 3), weight=1)
        
        ttk.Label(sa_frame, text="Initial Temperature:").grid(row=0, column=0, sticky=tk.E, pady=5, padx=5)
        self.sa_temp = ttk.Entry(sa_frame, width=10)
        self.sa_temp.insert(0, "100.0")
        self.sa_temp.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(sa_frame, text="Cooling Rate:").grid(row=1, column=0, sticky=tk.E, pady=5, padx=5)
        self.sa_cooling = ttk.Entry(sa_frame, width=10)
        self.sa_cooling.insert(0, "0.95")
        self.sa_cooling.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(sa_frame, text="Iterations:").grid(row=0, column=2, sticky=tk.E, pady=5, padx=5)
        self.sa_iterations = ttk.Entry(sa_frame, width=10)
        self.sa_iterations.insert(0, "1000")
        self.sa_iterations.grid(row=0, column=3, sticky=tk.W, pady=5)
        
        ttk.Label(sa_frame, text="Min Temperature:").grid(row=1, column=2, sticky=tk.E, pady=5, padx=5)
        self.sa_min_temp = ttk.Entry(sa_frame, width=10)
        self.sa_min_temp.insert(0, "0.01")
        self.sa_min_temp.grid(row=1, column=3, sticky=tk.W, pady=5)
       
        # Tabu Search Tab
        ts_frame = ttk.Frame(meta_notebook, padding=10)
        meta_notebook.add(ts_frame, text="Tabu Search")
        ts_frame.columnconfigure((0, 1, 2, 3), weight=1)
        

        ttk.Label(ts_frame, text="Iterations:").grid(row=0, column=2, sticky=tk.E, pady=5, padx=5)
        self.ts_iterations = ttk.Entry(ts_frame, width=10)
        self.ts_iterations.insert(0, "100")
        self.ts_iterations.grid(row=0, column=3, sticky=tk.W, pady=5)
        
        ttk.Label(ts_frame, text="Tabu Tenure:").grid(row=0, column=0, sticky=tk.E, pady=5, padx=5)
        self.ts_table_tenure = ttk.Entry(ts_frame, width=10)
        self.ts_table_tenure.insert(0, "20")
        self.ts_table_tenure.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(ts_frame, text="Frequency Memory:").grid(row=1, column=0, sticky=tk.E, pady=5, padx=5)
        self.freq_memory = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            ts_frame, 
            variable=self.freq_memory
        ).grid(row=1, column=1, sticky=tk.W, pady=5)

        ttk.Label(ts_frame, text="Aspiration Criteria:").grid(row=1, column=2, sticky=tk.E, pady=5, padx=5)
        self.ts_aspiration = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            ts_frame, 
            variable=self.ts_aspiration
        ).grid(row=1, column=3, sticky=tk.W, pady=5)

        # Genetic Algorithm Tab
        ga_frame = ttk.Frame(meta_notebook, padding=10)
        meta_notebook.add(ga_frame, text="Genetic Algorithm")
        ga_frame.columnconfigure((0, 1, 2, 3), weight=1)
        
        ttk.Label(ga_frame, text="Population Size:").grid(row=0, column=0, sticky=tk.E, pady=5, padx=5)
        self.ga_pop_size = ttk.Entry(ga_frame, width=10)
        self.ga_pop_size.insert(0, "50")
        self.ga_pop_size.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(ga_frame, text="Generations:").grid(row=0, column=2, sticky=tk.E, pady=5, padx=5)
        self.ga_generations = ttk.Entry(ga_frame, width=10)
        self.ga_generations.insert(0, "100")
        self.ga_generations.grid(row=0, column=3, sticky=tk.W, pady=5)
        
        ttk.Label(ga_frame, text="Mutation Rate:").grid(row=1, column=0, sticky=tk.E, pady=5, padx=5)
        self.ga_mutation = ttk.Entry(ga_frame, width=10)
        self.ga_mutation.insert(0, "0.1")
        self.ga_mutation.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(ga_frame, text="Crossover Rate:").grid(row=1, column=2, sticky=tk.E, pady=5, padx=5)
        self.ga_crossover = ttk.Entry(ga_frame, width=10)
        self.ga_crossover.insert(0, "0.8")
        self.ga_crossover.grid(row=1, column=3, sticky=tk.W, pady=5)
        
        ttk.Label(ga_frame, text="Elite Count:").grid(row=2, column=0, sticky=tk.E, pady=5, padx=5)
        self.ga_elite = ttk.Entry(ga_frame, width=10)
        self.ga_elite.insert(0, "2")
        self.ga_elite.grid(row=2, column=1, sticky=tk.W, pady=5)

        # Particle Swarm Tab
        pso_frame = ttk.Frame(meta_notebook, padding=10)
        meta_notebook.add(pso_frame, text="Particle Swarm")
        pso_frame.columnconfigure((0, 1, 2, 3), weight=1)

        ttk.Label(pso_frame, text="Particles:").grid(row=0, column=0, sticky=tk.E, pady=5, padx=5)
        self.pso_particles = ttk.Entry(pso_frame, width=10)
        self.pso_particles.insert(0, "30")
        self.pso_particles.grid(row=0, column=1, sticky=tk.W, pady=5)

        ttk.Label(pso_frame, text="Iterations:").grid(row=0, column=2, sticky=tk.E, pady=5, padx=5)
        self.pso_iterations = ttk.Entry(pso_frame, width=10)
        self.pso_iterations.insert(0, "100")
        self.pso_iterations.grid(row=0, column=3, sticky=tk.W, pady=5)

        ttk.Label(pso_frame, text="Inertia Weight:").grid(row=1, column=0, sticky=tk.E, pady=5, padx=5)
        self.pso_inertia = ttk.Entry(pso_frame, width=10)
        self.pso_inertia.insert(0, "0.5")
        self.pso_inertia.grid(row=1, column=1, sticky=tk.W, pady=5)

        ttk.Label(pso_frame, text="Cognitive Weight:").grid(row=1, column=2, sticky=tk.E, pady=5, padx=5)
        self.pso_cognitive = ttk.Entry(pso_frame, width=10)
        self.pso_cognitive.insert(0, "1.5")
        self.pso_cognitive.grid(row=1, column=3, sticky=tk.W, pady=5)

        ttk.Label(pso_frame, text="Social Weight:").grid(row=2, column=0, sticky=tk.E, pady=5, padx=5)
        self.pso_social = ttk.Entry(pso_frame, width=10)
        self.pso_social.insert(0, "1.5")
        self.pso_social.grid(row=2, column=1, sticky=tk.W, pady=5)

        # Differential Evolution Tab
        de_frame = ttk.Frame(meta_notebook, padding=10)
        meta_notebook.add(de_frame, text="Differential Evolution")
        de_frame.columnconfigure((0, 1, 2, 3), weight=1)

        ttk.Label(de_frame, text="Population Size:").grid(row=0, column=0, sticky=tk.E, pady=5, padx=5)
        self.de_population = ttk.Entry(de_frame, width=10)
        self.de_population.insert(0, "50")
        self.de_population.grid(row=0, column=1, sticky=tk.W, pady=5)

        ttk.Label(de_frame, text="Generations:").grid(row=0, column=2, sticky=tk.E, pady=5, padx=5)
        self.de_generations = ttk.Entry(de_frame, width=10)
        self.de_generations.insert(0, "100")
        self.de_generations.grid(row=0, column=3, sticky=tk.W, pady=5)

        ttk.Label(de_frame, text="F:").grid(row=1, column=0, sticky=tk.E, pady=5, padx=5)
        self.de_F = ttk.Entry(de_frame, width=10)
        self.de_F.insert(0, "0.8")
        self.de_F.grid(row=1, column=1, sticky=tk.W, pady=5)

        ttk.Label(de_frame, text="CR:").grid(row=1, column=2, sticky=tk.E, pady=5, padx=5)
        self.de_CR = ttk.Entry(de_frame, width=10)
        self.de_CR.insert(0, "0.9")
        self.de_CR.grid(row=1, column=3, sticky=tk.W, pady=5)

        # Optimal Solution Tab
        opt_frame = ttk.Frame(self.param_notebook, padding=10)
        self.param_notebook.add(opt_frame, text="Optimal Solution")
        opt_frame.columnconfigure((0, 1, 2, 3), weight=1)

        # Optimization type dropdown
        ttk.Label(opt_frame, text="Optimization Type:").grid(row=0, column=0, sticky=tk.E, pady=5, padx=5)
        self.opt_type = ttk.Combobox(
            opt_frame, 
            values=["sharpe", "min_variance", "utility", "target_return"],
            state="readonly"
        )
        self.opt_type.set("utility")
        self.opt_type.grid(row=0, column=1, sticky=tk.W, pady=5)

        # Target return (only shown when target_return is selected)
        self.target_return_label = ttk.Label(opt_frame, text="Target Return (%):")
        self.target_return_label.grid(row=1, column=0, sticky=tk.E, pady=5, padx=5)
        self.target_return_entry = ttk.Entry(opt_frame, width=10)
        self.target_return_entry.grid(row=1, column=1, sticky=tk.W, pady=5)
        self.target_return_label.grid_remove()
        self.target_return_entry.grid_remove()

        # Min weight constraint
        ttk.Label(opt_frame, text="Min Weight (%):").grid(row=2, column=0, sticky=tk.E, pady=5, padx=5)
        self.opt_min_weight = ttk.Entry(opt_frame, width=10)
        self.opt_min_weight.insert(0, "0")
        self.opt_min_weight.grid(row=2, column=1, sticky=tk.W, pady=5)

        self.opt_type.bind("<<ComboboxSelected>>", self.update_opt_parameters)

        # Visualization and controls
        viz_options_frame = ttk.Frame(main_frame)
        viz_options_frame.pack(fill=tk.X, pady=5)
        viz_options_frame.columnconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)

        ttk.Label(viz_options_frame, text="Visualization:").grid(row=0, column=0, sticky=tk.E, padx=5)
        self.viz_type = tk.StringVar(value="portfolio_allocation")
        viz_combobox = ttk.Combobox(
            viz_options_frame, 
            textvariable=self.viz_type,
            values=[
                "portfolio_allocation", 
                "efficient_frontier",
                "optimization_progress",
                "return_risk_comparison", 
                "correlation_matrix"
            ],
            width=20
        )
        viz_combobox.grid(row=0, column=1, sticky=tk.W, padx=5)

        pdf_button = ttk.Button(
            viz_options_frame, 
            text="Export to PDF", 
            command=self.export_to_pdf
        )
        pdf_button.grid(row=0, column=2, sticky=tk.E, padx=5)
        
        save_button = ttk.Button(
            viz_options_frame, 
            text="Save Results", 
            command=self.save_results
        )
        save_button.grid(row=0, column=3, sticky=tk.E, padx=5)
        
        run_button = ttk.Button(
            viz_options_frame, 
            text="Run Portfolio Optimization", 
            style="Accent.TButton", 
            command=self.run_analysis
        )
        run_button.grid(row=0, column=4, sticky=tk.E, padx=5)
        
        test_button = ttk.Button(
            viz_options_frame, 
            text="Test Portfolio Optimization", 
            style="Green.TButton",
            command=self.test_optimization
        )
        test_button.grid(row=0, column=5, sticky=tk.E, padx=5)
        
        self.test_all_algorithms = tk.BooleanVar(value=False)
        test_all_checkbox = ttk.Checkbutton(
            viz_options_frame,
            text="Test All Algorithms",
            variable=self.test_all_algorithms
        )
        test_all_checkbox.grid(row=0, column=6, sticky=tk.W, padx=5)
        
        viz_frame = ttk.LabelFrame(main_frame, text="Visualization", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        results_frame = ttk.LabelFrame(main_frame, text="Results Summary", padding=10)
        results_frame.pack(fill=tk.X, pady=5)

        results_text_frame = ttk.Frame(results_frame)
        results_text_frame.pack(fill=tk.X, expand=True)
        results_text_frame.columnconfigure(0, weight=1)
        
        self.results_text = tk.Text(results_text_frame, height=4, width=80)
        self.results_text.grid(row=0, column=0, sticky="ew")
        
        results_scrollbar = ttk.Scrollbar(results_text_frame, orient="vertical", command=self.results_text.yview)
        results_scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_text.config(yscrollcommand=results_scrollbar.set)

        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        status_frame.columnconfigure(0, weight=1)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)
        
        self.progress_var = tk.DoubleVar(value=0.0)
        progress_bar = ttk.Progressbar(
            status_frame, 
            orient="horizontal", 
            length=200, 
            mode="determinate", 
            variable=self.progress_var
        )
        progress_bar.pack(side=tk.RIGHT)

    def update_opt_parameters(self, event=None):
        """Show/hide target return field based on optimization type."""
        if self.opt_type.get() == "target_return":
            self.target_return_label.grid()
            self.target_return_entry.grid()
        else:
            self.target_return_label.grid_remove()
            self.target_return_entry.grid_remove()

    def browse_files(self):
        filenames = filedialog.askopenfilenames(
            title="Select CSV Files",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        
        if filenames:
            for filename in filenames:
                basename = os.path.basename(filename)
                existing_files = self.file_listbox.get(0, tk.END)
                if basename in existing_files:
                    messagebox.showinfo("Info", f"File {basename} is already loaded")
                    continue
                
                try:
                    self.current_data[basename] = pd.read_csv(filename)
                    self.file_listbox.insert(tk.END, basename)
                    self.status_var.set(f"Loaded {basename} successfully")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load file {basename}: {e}")
                    self.status_var.set("Error loading file")
    
    def browse_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder Containing CSV Files")
        
        if folder_path:
            for filename in os.listdir(folder_path):
                if filename.endswith(".csv"):
                    basename = os.path.basename(filename)
                    existing_files = self.file_listbox.get(0, tk.END)
                    if basename in existing_files:
                        messagebox.showinfo("Info", f"File {basename} is already loaded")
                        continue
                    
                    try:
                        file_path = os.path.join(folder_path, filename)
                        self.current_data[basename] = pd.read_csv(file_path)
                        self.file_listbox.insert(tk.END, basename)
                        self.status_var.set(f"Loaded {basename} successfully")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to load file {basename}: {e}")
                        self.status_var.set("Error loading file")
    
    def remove_selected_file(self):
        try:
            selected_indices = self.file_listbox.curselection()
            if selected_indices:
                indices = sorted(selected_indices, reverse=True)
                removed_files = []
                
                for idx in indices:
                    selected_file = self.file_listbox.get(idx)
                    self.file_listbox.delete(idx)
                    if selected_file in self.current_data:
                        del self.current_data[selected_file]
                    removed_files.append(selected_file)
                
                if removed_files:
                    if len(removed_files) == 1:
                        self.status_var.set(f"Removed {removed_files[0]}")
                    else:
                        self.status_var.set(f"Removed {len(removed_files)} files")
                
                if not self.current_data:
                    self.fig.clear()
                    self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove file: {e}")

    def remove_all_files(self):
        try:
            self.file_listbox.delete(0, tk.END)  
            self.current_data.clear()  
            self.status_var.set("All files removed successfully")
            
            self.fig.clear()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove all files: {e}")
    
    def show_selected_graph(self):
        """Show the price trends of selected stocks."""
        try:
            selected_indices = self.file_listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Warning", "Please select at least one file to display")
                return
            
            self.fig.clear()
            ax = self.fig.add_subplot(111)

            colors = plt.cm.tab10(range(len(selected_indices)))  
            
            for idx, selected_idx in enumerate(selected_indices):
                selected_file = self.file_listbox.get(selected_idx)
                data = self.current_data[selected_file]
                
                if 'Date' in data.columns and 'Close' in data.columns:
                    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
                        data['Date'] = pd.to_datetime(data['Date'])
                    
                    ax.plot(data['Date'], data['Close'], label=selected_file, color=colors[idx])
            
            ax.set_title("Stock Price History", fontsize=14, pad=20)  
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Price ($)", fontsize=12)
            ax.grid(True)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display graph: {e}")

    def preprocess_data(self):
        """Preprocess the loaded stock data for optimization."""
        if not self.current_data:
            messagebox.showwarning("Warning", "No data files selected")
            return None

        combined_data = pd.DataFrame()
        for stock_name, data in self.current_data.items():
            if 'Date' not in data.columns or 'Close' not in data.columns:
                messagebox.showerror("Error", f"Invalid data format in {stock_name}")
                return None

            data['Date'] = pd.to_datetime(data['Date'])
            data.dropna(subset=['Close'], inplace=True)
            data['Daily Return'] = data['Close'].pct_change()
            combined_data[stock_name] = data.set_index('Date')['Daily Return']

        combined_data.dropna(inplace=True)
        return combined_data
    
    def calculate_portfolio_metrics(self, weights, returns, cov_matrix):
        """Calculate portfolio return and risk."""
        portfolio_return = np.sum(returns.mean() * weights) * 252  
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        return portfolio_return, portfolio_risk, sharpe_ratio
    
    def test_optimization(self):
        """Run test optimization and display average metrics."""
        test_all = self.test_all_algorithms.get()
        
        if test_all:
            self.run_comparative_test()
        else:
            self.run_single_algorithm_test()

    def run_single_algorithm_test(self):
        """Test the currently selected algorithm multiple times."""
        returns_list = []
        risk_list = []
        sharpe_ratios_list = []
        weights_list = []
        execution_times = []  
        
        original_viz_type = self.viz_type.get()
        self.viz_type.set("None")
        no_iterations = 50
        
        selected_tab = self.param_notebook.tab(self.param_notebook.select(), "text")
        
        if selected_tab == "Metaheuristics":
            meta_frame = self.param_notebook.nametowidget(self.param_notebook.select())
            meta_notebook = meta_frame.winfo_children()[0]  
            selected_tab = meta_notebook.tab(meta_notebook.select(), "text")
        
        for i in range(no_iterations):
            try:
                self.status_var.set(f"Running {selected_tab} test {i+1}/{no_iterations}...")
                self.progress_var.set(int((i+1)/no_iterations * 100))
                self.root.update_idletasks()
                
                combined_data = self.preprocess_data()
                if combined_data is None:
                    continue
                
                returns = combined_data
                mu = returns.mean().to_numpy() * 252  
                cov_matrix = returns.cov().to_numpy() * 252

                if cov_matrix.shape[0] < 2:
                    continue
                
                allow_short = self.allow_short.get()
                risk_aversion = float(self.risk_aversion.get())
                max_weight = float(self.max_weight.get()) / 100.0 if not allow_short else 1.0
                
                start_time = time.time()
                
                if selected_tab == "Hill Climbing":
                    iterations = int(self.hc_iterations.get())
                    step_size = float(self.hc_step.get())
                    restarts = int(self.hc_restarts.get())
                    optimal_weights, _ = hill_climbing(
                        mu, cov_matrix, iterations, step_size, 
                        risk_aversion=risk_aversion,
                        allow_short=allow_short,
                        max_weight=max_weight,
                        restarts=restarts
                    )
                    
                elif selected_tab == "Simulated Annealing":
                    initial_temp = float(self.sa_temp.get())
                    cooling_rate = float(self.sa_cooling.get())
                    iterations = int(self.sa_iterations.get())
                    min_temp = float(self.sa_min_temp.get())
                
                    optimal_weights, _ = simulated_annealing(
                        expected_returns=mu,
                        cov_matrix=cov_matrix,
                        initial_temp=initial_temp,
                        cooling_rate=cooling_rate,
                        iterations=iterations,
                        risk_aversion=risk_aversion,
                        allow_short=allow_short,
                        max_weight=max_weight,
                        min_temp=min_temp
                    )

                elif selected_tab == "Tabu Search":
                    iterations = int(self.ts_iterations.get())
                    ts_table_tenure = int(self.ts_table_tenure.get())
                    aspiration = bool(self.ts_aspiration.get())
                    freq_memory = bool(self.freq_memory.get())
                    optimal_weights, _ = tabu_search(
                        mu,  
                        cov_matrix,       
                        iterations=iterations,
                        tabu_tenure=ts_table_tenure,
                        frequency_memory=freq_memory,
                        aspiration_criteria=aspiration,
                        step_size=0.05,         
                        risk_aversion=risk_aversion,      
                        allow_short=allow_short,      
                        max_weight=max_weight,         
                        restarts=5              
                    )
                                    
                elif selected_tab == "Genetic Algorithm":
                    population_size = int(self.ga_pop_size.get())
                    generations = int(self.ga_generations.get())
                    mutation_rate = float(self.ga_mutation.get())
                    crossover_rate = float(self.ga_crossover.get())
                    elite_count = int(self.ga_elite.get())

                    optimal_weights, _ = genetic_algorithm(
                        mu, cov_matrix, population_size, generations, 
                        mutation_rate, crossover_rate,
                        risk_aversion=risk_aversion,
                        allow_short=allow_short,
                        max_weight=max_weight,
                        elite_count=elite_count
                    )
                    
                elif selected_tab == "Particle Swarm":
                    n_particles = int(self.pso_particles.get())
                    iterations = int(self.pso_iterations.get())
                    inertia_weight = float(self.pso_inertia.get())
                    cognitive_weight = float(self.pso_cognitive.get())
                    social_weight = float(self.pso_social.get())
                    
                    optimal_weights, _ = particle_swarm_optimization(
                        expected_returns=mu,
                        cov_matrix=cov_matrix,
                        n_particles=n_particles,
                        iterations=iterations,
                        inertia_weight=inertia_weight,
                        cognitive_weight=cognitive_weight,
                        social_weight=social_weight,
                        risk_aversion=risk_aversion,
                        allow_short=allow_short,
                        max_weight=max_weight
                    )
                    
                elif selected_tab == "Differential Evolution":
                    population_size = int(self.de_population.get())
                    generations = int(self.de_generations.get())
                    F = float(self.de_F.get())
                    CR = float(self.de_CR.get())
                    
                    optimal_weights, _ = differential_evolution(
                        expected_returns=mu,
                        cov_matrix=cov_matrix,
                        population_size=population_size,
                        generations=generations,
                        F=F,
                        CR=CR,
                        risk_aversion=risk_aversion,
                        allow_short=allow_short,
                        max_weight=max_weight
                    )
                
                elif selected_tab == "Optimal Solution":
                    opt_type = self.opt_type.get()
                    target_return = None
                    if opt_type == "target_return":
                        try:
                            target_return = float(self.target_return_entry.get()) / 100.0
                        except ValueError:
                            continue
                    
                    optimal_weights, _ = exact_optimization(
                        expected_returns=mu,
                        cov_matrix=cov_matrix,
                        optimization_type=opt_type,
                        target_return=target_return,
                        risk_aversion=risk_aversion,
                        allow_short=allow_short,
                        max_weight=max_weight,
                        min_weight=0.0
                    )

                else:
                    continue  
                
                end_time = time.time()
                execution_time = end_time - start_time
                execution_times.append(execution_time)
                
                portfolio_return, portfolio_risk, sharpe_ratio = self.calculate_portfolio_metrics(
                    optimal_weights, returns, returns.cov()
                )
                
                returns_list.append(portfolio_return)
                risk_list.append(portfolio_risk)
                sharpe_ratios_list.append(sharpe_ratio)
                weights_list.append(optimal_weights)
                
            except Exception as e:
                print(f"Iteration {i+1} failed: {str(e)}")
                continue
    
        self.display_test_results(selected_tab, no_iterations, returns_list, risk_list, 
                                sharpe_ratios_list, execution_times)
        self.viz_type.set(original_viz_type)

    def run_comparative_test(self):
        """Test all algorithms with their default parameters and compare results."""
        algorithm_results = {}
        no_iterations = 5  
        
        original_viz_type = self.viz_type.get()
        self.viz_type.set("None")
        
        algorithms = [
            ("Hill Climbing", self.run_hill_climbing_test),
            ("Simulated Annealing", self.run_simulated_annealing_test),
            ("Tabu Search", self.run_tabu_search_test),
            ("Genetic Algorithm", self.run_genetic_algorithm_test),
            ("Particle Swarm", self.run_particle_swarm_test),
            ("Differential Evolution", self.run_differential_evolution_test),
            ("Exact Optimization", self.run_exact_optimization_test),
        ]
        
        total_algorithms = len(algorithms)
        
        for idx, (algo_name, test_function) in enumerate(algorithms):
            try:
                self.status_var.set(f"Testing {algo_name} ({idx+1}/{total_algorithms})...")
                self.progress_var.set(int((idx+1)/total_algorithms * 100))
                self.root.update_idletasks()
                
                returns_list, risk_list, sharpe_list = test_function(no_iterations)
                
                if returns_list:  
                    algorithm_results[algo_name] = {
                        'returns': returns_list,
                        'risks': risk_list,
                        'sharpes': sharpe_list
                    }
                    
            except Exception as e:
                print(f"Algorithm {algo_name} failed: {str(e)}")
                continue
        
        self.display_comparative_results(algorithm_results)
        self.viz_type.set(original_viz_type)

    def run_hill_climbing_test(self, no_iterations):
        """Run hill climbing test with default parameters."""
        returns_list = []
        risk_list = []
        sharpe_list = []
        
        for _ in range(no_iterations):
            combined_data = self.preprocess_data()
            if combined_data is None:
                continue
                
            returns = combined_data
            mu = returns.mean().to_numpy() * 252  
            cov_matrix = returns.cov().to_numpy() * 252

            if cov_matrix.shape[0] < 2:
                continue
                
            allow_short = self.allow_short.get()
            risk_aversion = float(self.risk_aversion.get())
            max_weight = float(self.max_weight.get()) / 100.0 if not allow_short else 1.0
            
            optimal_weights, _ = hill_climbing(
                mu, cov_matrix, 
                iterations=1000,
                step_size=0.05,
                risk_aversion=risk_aversion,
                allow_short=allow_short,
                max_weight=max_weight,
                restarts=5
            )
            
            portfolio_return, portfolio_risk, sharpe_ratio = self.calculate_portfolio_metrics(
                optimal_weights, returns, returns.cov()
            )
            
            returns_list.append(portfolio_return)
            risk_list.append(portfolio_risk)
            sharpe_list.append(sharpe_ratio)
            
        return returns_list, risk_list, sharpe_list

    def run_simulated_annealing_test(self, no_iterations):
        """Run simulated annealing test with default parameters."""
        returns_list = []
        risk_list = []
        sharpe_list = []
        
        for _ in range(no_iterations):
            combined_data = self.preprocess_data()
            if combined_data is None:
                continue
                
            returns = combined_data
            mu = returns.mean().to_numpy() * 252  
            cov_matrix = returns.cov().to_numpy() * 252

            if cov_matrix.shape[0] < 2:
                continue
                
            allow_short = self.allow_short.get()
            risk_aversion = float(self.risk_aversion.get())
            max_weight = float(self.max_weight.get()) / 100.0 if not allow_short else 1.0
            
            optimal_weights, _ = simulated_annealing(
                expected_returns=mu,
                cov_matrix=cov_matrix,
                initial_temp=100.0,
                cooling_rate=0.95,
                iterations=1000,
                risk_aversion=risk_aversion,
                allow_short=allow_short,
                max_weight=max_weight,
                min_temp=0.01
            )
            
            portfolio_return, portfolio_risk, sharpe_ratio = self.calculate_portfolio_metrics(
                optimal_weights, returns, returns.cov()
            )
            
            returns_list.append(portfolio_return)
            risk_list.append(portfolio_risk)
            sharpe_list.append(sharpe_ratio)
            
        return returns_list, risk_list, sharpe_list

    def run_tabu_search_test(self, no_iterations):
        """Run tabu search test with default parameters."""
        returns_list = []
        risk_list = []
        sharpe_list = []
        
        for _ in range(no_iterations):
            combined_data = self.preprocess_data()
            if combined_data is None:
                continue
                
            returns = combined_data
            mu = returns.mean().to_numpy() * 252  
            cov_matrix = returns.cov().to_numpy() * 252

            if cov_matrix.shape[0] < 2:
                continue
                
            allow_short = self.allow_short.get()
            risk_aversion = float(self.risk_aversion.get())
            max_weight = float(self.max_weight.get()) / 100.0 if not allow_short else 1.0
            
            optimal_weights, _ = tabu_search(
                mu,  
                cov_matrix,       
                iterations=100,
                tabu_tenure=20,
                frequency_memory=True,
                aspiration_criteria=True,
                step_size=0.05,         
                risk_aversion=risk_aversion,      
                allow_short=allow_short,      
                max_weight=max_weight,         
                restarts=5              
            )
            
            portfolio_return, portfolio_risk, sharpe_ratio = self.calculate_portfolio_metrics(
                optimal_weights, returns, returns.cov()
            )
            
            returns_list.append(portfolio_return)
            risk_list.append(portfolio_risk)
            sharpe_list.append(sharpe_ratio)
            
        return returns_list, risk_list, sharpe_list

    def run_genetic_algorithm_test(self, no_iterations):
        """Run genetic algorithm test with default parameters."""
        returns_list = []
        risk_list = []
        sharpe_list = []
        
        for _ in range(no_iterations):
            combined_data = self.preprocess_data()
            if combined_data is None:
                continue
                
            returns = combined_data
            mu = returns.mean().to_numpy() * 252  
            cov_matrix = returns.cov().to_numpy() * 252

            if cov_matrix.shape[0] < 2:
                continue
                
            allow_short = self.allow_short.get()
            risk_aversion = float(self.risk_aversion.get())
            max_weight = float(self.max_weight.get()) / 100.0 if not allow_short else 1.0
            
            optimal_weights, _ = genetic_algorithm(
                mu, cov_matrix, 
                population_size=50,
                generations=100,
                mutation_rate=0.1,
                crossover_rate=0.8,
                risk_aversion=risk_aversion,
                allow_short=allow_short,
                max_weight=max_weight,
                elite_count=2
            )
            
            portfolio_return, portfolio_risk, sharpe_ratio = self.calculate_portfolio_metrics(
                optimal_weights, returns, returns.cov()
            )
            
            returns_list.append(portfolio_return)
            risk_list.append(portfolio_risk)
            sharpe_list.append(sharpe_ratio)
            
        return returns_list, risk_list, sharpe_list

    def run_particle_swarm_test(self, no_iterations):
        """Run particle swarm test with default parameters."""
        returns_list = []
        risk_list = []
        sharpe_list = []
        
        for _ in range(no_iterations):
            combined_data = self.preprocess_data()
            if combined_data is None:
                continue
                
            returns = combined_data
            mu = returns.mean().to_numpy() * 252  
            cov_matrix = returns.cov().to_numpy() * 252

            if cov_matrix.shape[0] < 2:
                continue
                
            allow_short = self.allow_short.get()
            risk_aversion = float(self.risk_aversion.get())
            max_weight = float(self.max_weight.get()) / 100.0 if not allow_short else 1.0
            
            optimal_weights, _ = particle_swarm_optimization(
                expected_returns=mu,
                cov_matrix=cov_matrix,
                n_particles=30,
                iterations=100,
                inertia_weight=0.5,
                cognitive_weight=1.5,
                social_weight=1.5,
                risk_aversion=risk_aversion,
                allow_short=allow_short,
                max_weight=max_weight
            )
            
            portfolio_return, portfolio_risk, sharpe_ratio = self.calculate_portfolio_metrics(
                optimal_weights, returns, returns.cov()
            )
            
            returns_list.append(portfolio_return)
            risk_list.append(portfolio_risk)
            sharpe_list.append(sharpe_ratio)
            
        return returns_list, risk_list, sharpe_list

    def run_differential_evolution_test(self, no_iterations):
        """Run differential evolution test with default parameters."""
        returns_list = []
        risk_list = []
        sharpe_list = []
        
        for _ in range(no_iterations):
            combined_data = self.preprocess_data()
            if combined_data is None:
                continue
                
            returns = combined_data
            mu = returns.mean().to_numpy() * 252  
            cov_matrix = returns.cov().to_numpy() * 252

            if cov_matrix.shape[0] < 2:
                continue
                
            allow_short = self.allow_short.get()
            risk_aversion = float(self.risk_aversion.get())
            max_weight = float(self.max_weight.get()) / 100.0 if not allow_short else 1.0
            
            optimal_weights, _ = differential_evolution(
                expected_returns=mu,
                cov_matrix=cov_matrix,
                population_size=50,
                generations=100,
                F=0.8,
                CR=0.9,
                risk_aversion=risk_aversion,
                allow_short=allow_short,
                max_weight=max_weight
            )
            
            portfolio_return, portfolio_risk, sharpe_ratio = self.calculate_portfolio_metrics(
                optimal_weights, returns, returns.cov()
            )
            
            returns_list.append(portfolio_return)
            risk_list.append(portfolio_risk)
            sharpe_list.append(sharpe_ratio)
            
        return returns_list, risk_list, sharpe_list

    def run_exact_optimization_test(self, no_iterations):
        """Run exact optimization test with default parameters."""
        returns_list = []
        risk_list = []
        sharpe_list = []
        
        for _ in range(no_iterations):
            combined_data = self.preprocess_data()
            if combined_data is None:
                continue
                
            returns = combined_data
            mu = returns.mean().to_numpy() * 252
            cov_matrix = returns.cov().to_numpy() * 252
            
            allow_short = self.allow_short.get()
            max_weight = float(self.max_weight.get()) / 100.0
            min_weight = 0.0
            
            optimal_weights, _ = exact_optimization(
                expected_returns=mu,
                cov_matrix=cov_matrix,
                optimization_type='utility',
                risk_aversion=float(self.risk_aversion.get()),
                allow_short=allow_short,
                max_weight=max_weight,
                min_weight=min_weight
            )
            
            portfolio_return, portfolio_risk, sharpe_ratio = self.calculate_portfolio_metrics(
                optimal_weights, returns, returns.cov()
            )
            
            returns_list.append(portfolio_return)
            risk_list.append(portfolio_risk)
            sharpe_list.append(sharpe_ratio)
            
        return returns_list, risk_list, sharpe_list

    def display_test_results(self, algorithm_name, no_iterations, returns_list, risk_list, sharpe_list, execution_times=None):
        """Display results for a single algorithm test including execution times."""
        if not returns_list:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "No valid test results were obtained.")
            return
            
        avg_return = np.mean(returns_list) * 100
        avg_risk = np.mean(risk_list) * 100
        avg_sharpe = np.mean(sharpe_list)
        return_std = np.std(returns_list) * 100
        risk_std = np.std(risk_list) * 100
        sharpe_std = np.std(sharpe_list)
        
        result_summary = (
            f"{algorithm_name} Test Results ({no_iterations} runs):\n\n"
            f"Average Annual Return: {avg_return:.4f}% ± {return_std:.4f}\n"
            f"Average Annual Risk: {avg_risk:.4f}% ± {risk_std:.4f}\n"
            f"Average Sharpe Ratio: {avg_sharpe:.4f} ± {sharpe_std:.4f}\n\n"
            f"Best Return: {np.max(returns_list)*100:.4f}%\n"
            f"Best Risk: {np.min(risk_list)*100:.4f}%\n"
            f"Best Sharpe: {np.max(sharpe_list):.4f}\n\n"
            f"Worst Return: {np.min(returns_list)*100:.4f}%\n"
            f"Worst Risk: {np.max(risk_list)*100:.4f}%\n"
            f"Worst Sharpe: {np.min(sharpe_list):.4f}"
        )
        
        if execution_times and len(execution_times) > 0:
            avg_time = np.mean(execution_times)
            min_time = np.min(execution_times)
            max_time = np.max(execution_times)
            total_time = np.sum(execution_times)
            
            time_summary = (
                "\n\n--- Execution Time Statistics ---\n"
                f"Average per run: {avg_time:.4f} seconds\n"
                f"Fastest run: {min_time:.4f} seconds\n"
                f"Slowest run: {max_time:.4f} seconds\n"
                f"Total time: {total_time:.4f} seconds"
            )
            result_summary += time_summary
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, result_summary)
        self.status_var.set(f"{algorithm_name} test completed with {no_iterations} runs")

    def display_comparative_results(self, algorithm_results):
        """Display comparative results for all algorithms."""
        if not algorithm_results:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "No valid test results were obtained.")
            return
            
        results_text = "Algorithm Comparison Results:\n\n"
        results_text += "{:<25} {:<15} {:<15} {:<15}\n".format(
            "Algorithm", "Avg Return", "Avg Risk", "Avg Sharpe"
        )
        results_text += "-"*70 + "\n"
        
        algo_names = []
        avg_returns = []
        avg_risks = []
        avg_sharpes = []
        
        for algo_name, results in algorithm_results.items():
            avg_return = np.mean(results['returns']) * 100
            avg_risk = np.mean(results['risks']) * 100
            avg_sharpe = np.mean(results['sharpes'])
            
            results_text += "{:<25} {:<15.2f} {:<15.2f} {:<15.2f}\n".format(
                algo_name, avg_return, avg_risk, avg_sharpe
            )
            
            algo_names.append(algo_name)
            avg_returns.append(avg_return)
            avg_risks.append(avg_risk)
            avg_sharpes.append(avg_sharpe)
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results_text)
        
        self.fig.clear()
        
        ax1 = self.fig.add_subplot(311)
        ax2 = self.fig.add_subplot(312)
        ax3 = self.fig.add_subplot(313)
        
        ax1.bar(algo_names, avg_returns, color='skyblue')
        ax1.set_title('Average Annual Return (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(algo_names, avg_risks, color='lightcoral')
        ax2.set_title('Average Annual Risk (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        ax3.bar(algo_names, avg_sharpes, color='lightgreen')
        ax3.set_title('Average Sharpe Ratio')
        ax3.tick_params(axis='x', rotation=45)
        
        self.fig.tight_layout()
        self.canvas.draw()
        self.status_var.set(f"Comparative test completed with {len(algorithm_results)} algorithms")

    def run_analysis(self):
        """Run portfolio optimization based on selected algorithm category."""
        selected_tab = self.param_notebook.tab(self.param_notebook.select(), "text")
        
        if selected_tab == "Metaheuristics":
            self.run_metaheuristic_optimization()
        elif selected_tab == "Optimal Solution":
            self.run_exact_optimization()
        else:
            messagebox.showwarning("Warning", "Please select an algorithm category")

    def run_metaheuristic_optimization(self):
        """Run one of the metaheuristic algorithms."""
        selected_algo = self.param_notebook.nametowidget(
            self.param_notebook.select()
        ).winfo_children()[0].tab(
            self.param_notebook.nametowidget(
                self.param_notebook.select()
            ).winfo_children()[0].select(), "text"
        )
        
        try:
            combined_data = self.preprocess_data()
            if combined_data is None:
                return
                
            returns = combined_data
            mu = returns.mean().to_numpy() * 252  
            cov_matrix = returns.cov().to_numpy() * 252

            if cov_matrix.shape[0] < 2:
                messagebox.showerror("Error", "At least two assets are required for optimization.")
                return
            
            allow_short = self.allow_short.get()
            risk_aversion = float(self.risk_aversion.get())
            max_weight = float(self.max_weight.get()) / 100.0 if not allow_short else 1.0
            
            self.status_var.set(f"Running {selected_algo} optimization...")
            self.progress_var.set(30)
            self.root.update_idletasks()
            
            if selected_algo == "Hill Climbing":
                iterations = int(self.hc_iterations.get())
                step_size = float(self.hc_step.get())
                restarts = int(self.hc_restarts.get())
                optimal_weights, optimization_history = hill_climbing(
                    mu, cov_matrix, iterations, step_size, 
                    risk_aversion=risk_aversion,
                    allow_short=allow_short,
                    max_weight=max_weight,
                    restarts=restarts
                )
                
            elif selected_algo == "Simulated Annealing":
                initial_temp = float(self.sa_temp.get())
                cooling_rate = float(self.sa_cooling.get())
                iterations = int(self.sa_iterations.get())
                min_temp = float(self.sa_min_temp.get())
            
                optimal_weights, optimization_history = simulated_annealing(
                    expected_returns=mu,
                    cov_matrix=cov_matrix,
                    initial_temp=initial_temp,
                    cooling_rate=cooling_rate,
                    iterations=iterations,
                    risk_aversion=risk_aversion,
                    allow_short=allow_short,
                    max_weight=max_weight,
                    min_temp=min_temp
                )

            elif selected_algo == "Tabu Search":
                iterations = int(self.ts_iterations.get())
                ts_table_tenure = int(self.ts_table_tenure.get())
                aspiration = bool(self.ts_aspiration.get())
                freq_memory = bool(self.freq_memory.get())
                optimal_weights, optimization_history = tabu_search(
                    mu,  
                    cov_matrix,       
                    iterations=iterations,
                    tabu_tenure=ts_table_tenure,
                    frequency_memory=freq_memory,
                    aspiration_criteria=aspiration,
                    step_size=0.05,         
                    risk_aversion=risk_aversion,      
                    allow_short=allow_short,      
                    max_weight=max_weight,         
                    restarts=5              
                )
                                    
            elif selected_algo == "Genetic Algorithm":
                population_size = int(self.ga_pop_size.get())
                generations = int(self.ga_generations.get())
                mutation_rate = float(self.ga_mutation.get())
                crossover_rate = float(self.ga_crossover.get())
                elite_count = int(self.ga_elite.get())

                optimal_weights, optimization_history = genetic_algorithm(
                    mu, cov_matrix, population_size, generations, 
                    mutation_rate, crossover_rate,
                    risk_aversion=risk_aversion,
                    allow_short=allow_short,
                    max_weight=max_weight,
                    elite_count=elite_count
                )
                
            elif selected_algo == "Particle Swarm":
                n_particles = int(self.pso_particles.get())
                iterations = int(self.pso_iterations.get())
                inertia_weight = float(self.pso_inertia.get())
                cognitive_weight = float(self.pso_cognitive.get())
                social_weight = float(self.pso_social.get())
                
                optimal_weights, optimization_history = particle_swarm_optimization(
                    expected_returns=mu,
                    cov_matrix=cov_matrix,
                    n_particles=n_particles,
                    iterations=iterations,
                    inertia_weight=inertia_weight,
                    cognitive_weight=cognitive_weight,
                    social_weight=social_weight,
                    risk_aversion=risk_aversion,
                    allow_short=allow_short,
                    max_weight=max_weight
                )
                
            elif selected_algo == "Differential Evolution":
                population_size = int(self.de_population.get())
                generations = int(self.de_generations.get())
                F = float(self.de_F.get())
                CR = float(self.de_CR.get())
                
                optimal_weights, optimization_history = differential_evolution(
                    expected_returns=mu,
                    cov_matrix=cov_matrix,
                    population_size=population_size,
                    generations=generations,
                    F=F,
                    CR=CR,
                    risk_aversion=risk_aversion,
                    allow_short=allow_short,
                    max_weight=max_weight
                )

            else:
                messagebox.showwarning("Warning", "Algorithm not implemented")
                return
            
            self.process_optimization_results(
                optimal_weights, 
                optimization_history, 
                returns, 
                cov_matrix, 
                selected_algo
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Optimization failed: {str(e)}")
            self.status_var.set("Optimization failed")

    def run_exact_optimization(self):
        """Run exact optimization algorithm from the Optimal Solution tab."""
        try:
            combined_data = self.preprocess_data()
            if combined_data is None:
                return
                
            returns = combined_data
            mu = returns.mean().to_numpy() * 252
            cov_matrix = returns.cov().to_numpy() * 252
            
            allow_short = self.allow_short.get()
            max_weight = float(self.max_weight.get()) / 100.0
            min_weight = float(self.opt_min_weight.get()) / 100.0
            opt_type = self.opt_type.get()
            
            target_return = None
            if opt_type == "target_return":
                try:
                    target_return = float(self.target_return_entry.get()) / 100.0
                except ValueError:
                    messagebox.showerror("Error", "Invalid target return value")
                    return
            
            self.status_var.set("Running Exact Optimization...")
            self.progress_var.set(30)
            self.root.update_idletasks()
            
            optimal_weights, results = exact_optimization(
                expected_returns=mu,
                cov_matrix=cov_matrix,
                optimization_type=opt_type,
                target_return=target_return,
                risk_aversion=float(self.risk_aversion.get()),
                allow_short=allow_short,
                max_weight=max_weight,
                min_weight=min_weight
            )
            
            portfolio_return = np.sum(returns.mean() * optimal_weights) * 252
            portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(returns.cov() * 252, optimal_weights)))
            
            self.optimization_results = {
                'weights': optimal_weights,
                'stock_names': list(returns.columns),
                'return': portfolio_return,
                'risk': portfolio_risk,
                'sharpe_ratio': portfolio_return / portfolio_risk,
                'algorithm': f"Exact Optimization ({opt_type})",
                'efficient_frontier': results.get('efficient_frontier', None)
            }
            
            self.display_visualization()
            self.display_results_summary()
            self.progress_var.set(100)
            self.status_var.set(f"Exact optimization complete. Sharpe Ratio: {self.optimization_results['sharpe_ratio']:.4f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Exact optimization failed: {str(e)}")
            self.status_var.set("Exact optimization failed")

    def process_optimization_results(self, optimal_weights, optimization_history, returns, cov_matrix, algorithm):
        """Common method to process results from any optimization algorithm."""
        portfolio_return = np.sum(returns.mean() * optimal_weights) * 252
        portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(returns.cov() * 252, optimal_weights)))
        
        self.optimization_results = {
            'weights': optimal_weights,
            'stock_names': list(returns.columns),
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': portfolio_return / portfolio_risk,
            'history': optimization_history,
            'algorithm': algorithm,
            'returns_data': returns,
            'cov_matrix': cov_matrix
        }
        
        self.display_visualization()
        self.display_results_summary()
        self.progress_var.set(100)
        self.status_var.set(f"{algorithm} optimization complete. Sharpe Ratio: {self.optimization_results['sharpe_ratio']:.4f}")

    def display_visualization(self):
        """Display visualization based on selected type."""
        if not self.optimization_results:
            return
        
        viz_type = self.viz_type.get()
        self.fig.clear()
        
        if viz_type == "portfolio_allocation":
            self.plot_portfolio_allocation()
        elif viz_type == "efficient_frontier":
            self.plot_efficient_frontier()
        elif viz_type == "optimization_progress":
            self.plot_optimization_progress()
        elif viz_type == "return_risk_comparison":
            self.plot_return_risk_comparison()
        elif viz_type == "correlation_matrix":
            self.plot_correlation_matrix()
        
        self.canvas.draw()
    
    def plot_portfolio_allocation(self):
        """Plot portfolio allocation pie chart."""
        weights = self.optimization_results['weights']
        stock_names = self.optimization_results['stock_names']
        
        threshold = 0.005 
        filtered_weights = []
        filtered_names = []
        other_weight = 0
        
        for w, name in zip(weights, stock_names):
            if w > threshold:
                filtered_weights.append(w)
                filtered_names.append(name)
            else:
                other_weight += w
        
        if other_weight > 0:
            filtered_weights.append(other_weight)
            filtered_names.append("Other")
        ax = self.fig.add_subplot(111)
        
        colors = plt.cm.tab20(range(len(filtered_weights)))
        
        wedges, texts, autotexts = ax.pie(
            filtered_weights, 
            labels=filtered_names, 
            autopct='%1.1f%%',
            startangle=90, 
            colors=colors,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
        
        ax.set_title(f"Portfolio Allocation - {self.optimization_results['algorithm']}", fontsize=14)
        
        investment = float(self.initial_investment.get())
        portfolio_return = self.optimization_results['return']
        portfolio_risk = self.optimization_results['risk']
        
        details = (
            f"Initial Investment: ${investment:,.2f}\n"
            f"Expected Annual Return: {portfolio_return*100:.2f}%\n"
            f"Expected Annual Risk: {portfolio_risk*100:.2f}%\n"
            f"Sharpe Ratio: {self.optimization_results['sharpe_ratio']:.4f}"
        )
        
        self.fig.text(0.02, 0.02, details, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    def plot_efficient_frontier(self):
        """Plot the efficient frontier with current portfolio."""
        returns_data = self.optimization_results['returns_data']
        cov_matrix = self.optimization_results['cov_matrix']
        current_return = self.optimization_results['return']
        current_risk = self.optimization_results['risk']
        
        num_portfolios = 1000
        results = np.zeros((4, num_portfolios))
        
        for i in range(num_portfolios):
            try:
                weights = np.random.random(len(returns_data.columns))
                if np.sum(weights) == 0:
                    raise ValueError("Invalid weight distribution")
                weights /= np.sum(weights)
            except Exception as e:
                messagebox.showerror("Error", f"Weight normalization failed: {e}")
                return
            
            portfolio_return = np.sum(returns_data.mean() * 252 * weights)
            portfolio_risk = np.sqrt(
                np.dot(weights.T, np.dot(returns_data.cov() * 252, weights))
            )
            sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            
            results[0, i] = portfolio_risk
            results[1, i] = portfolio_return
            results[2, i] = sharpe_ratio
            
        ax = self.fig.add_subplot(111)
        scatter = ax.scatter(
            results[0, :], 
            results[1, :], 
            c=results[2, :], 
            s=10, 
            alpha=0.3,
            cmap='viridis'
        )
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sharpe Ratio')
        
        ax.scatter(
            current_risk, 
            current_return, 
            s=200, 
            c='red', 
            marker='*', 
            label='Optimized Portfolio'
        )
        
        for i, stock in enumerate(returns_data.columns):
            stock_return = returns_data.mean()[i] * 252
            stock_risk = np.sqrt(cov_matrix[i, i])
            ax.scatter(
                stock_risk, 
                stock_return, 
                s=100, 
                marker='o', 
                label=stock
            )
        
        ax.set_title("Efficient Frontier", fontsize=14)
        ax.set_xlabel("Expected Volatility (Risk)", fontsize=12)
        ax.set_ylabel("Expected Return", fontsize=12)
        ax.grid(True)
        
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
        
    def plot_optimization_progress(self):
        """Plot the optimization progress over iterations."""
        history = self.optimization_results['history']
        algorithm = self.optimization_results['algorithm']
        
        if not history:
            messagebox.showwarning("Warning", "No optimization history available for plotting")
            return
        
        ax = self.fig.add_subplot(111)
        
        iterations = list(range(len(history)))
        utility_values = [record.get('utility', 0) for record in history]
        
        ax.plot(iterations, utility_values, '-o', markersize=3, alpha=0.7)
        
        best_utilities = np.maximum.accumulate(utility_values)
        ax.plot(iterations, best_utilities, 'r-', alpha=0.5, label='Best Found')
        
        ax.set_title(f"{algorithm} Optimization Progress", fontsize=14)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Utility Value", fontsize=12)
        ax.grid(True)
        ax.legend()
        
        best_iteration = np.argmax(utility_values)
        best_utility = utility_values[best_iteration]
        
        ax.annotate(
            f'Best: {best_utility:.4f}',
            xy=(best_iteration, best_utility),
            xytext=(best_iteration, best_utility*0.9),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=7),
            fontsize=9
        )
    
    def plot_return_risk_comparison(self):
        """Plot return vs risk for each asset and the optimized portfolio."""
        returns_data = self.optimization_results['returns_data']
        weights = self.optimization_results['weights']
        stock_names = self.optimization_results['stock_names']
        portfolio_return = self.optimization_results['return']
        portfolio_risk = self.optimization_results['risk']
        
        annual_returns = returns_data.mean() * 252
        annual_risks = returns_data.std() * np.sqrt(252)
        
        ax = self.fig.add_subplot(111)
        
        ax.scatter(
            annual_risks, 
            annual_returns, 
            s=100, 
            alpha=0.7, 
            label='Individual Assets'
        )
        
        for i, stock in enumerate(stock_names):
            weight = weights[i]
            ax.annotate(
                f"{stock} ({weight*100:.1f}%)",
                xy=(annual_risks[i], annual_returns[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        ax.scatter(
            portfolio_risk, 
            portfolio_return, 
            s=200, 
            c='red', 
            marker='*', 
            label='Optimized Portfolio'
        )
        
        equal_weights = np.ones(len(stock_names)) / len(stock_names)
        equal_return = np.sum(annual_returns * equal_weights)
        equal_risk = np.sqrt(
            np.dot(equal_weights.T, np.dot(returns_data.cov() * 252, equal_weights))
        )
        
        ax.scatter(
            equal_risk, 
            equal_return, 
            s=150, 
            c='green', 
            marker='s', 
            label='Equal Weight Portfolio'
        )
        
        ax.set_title("Return vs Risk Comparison", fontsize=14)
        ax.set_xlabel("Expected Risk (Volatility)", fontsize=12)
        ax.set_ylabel("Expected Return", fontsize=12)
        ax.grid(True)
        ax.legend()
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix of assets."""
        returns_data = self.optimization_results['returns_data']
        
        corr_matrix = returns_data.corr()
        
        ax = self.fig.add_subplot(111)
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        cbar = plt.colorbar(im)
        cbar.set_label('Correlation Coefficient')
        
        tick_labels = corr_matrix.columns
        ax.set_xticks(np.arange(len(tick_labels)))
        ax.set_yticks(np.arange(len(tick_labels)))
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_yticklabels(tick_labels)
        
        for i in range(len(tick_labels)):
            for j in range(len(tick_labels)):
                text = ax.text(
                    j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                    ha="center", va="center", 
                    color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white",
                    fontsize=8
                )
        
        ax.set_title("Asset Correlation Matrix", fontsize=14)
    
    def display_results_summary(self):
        """Display a summary of optimization results in the text area."""
        if not self.optimization_results:
            return
        
        self.results_text.delete(1.0, tk.END)
        
        weights = self.optimization_results['weights']
        stock_names = self.optimization_results['stock_names']
        portfolio_return = self.optimization_results['return']
        portfolio_risk = self.optimization_results['risk']
        sharpe_ratio = self.optimization_results['sharpe_ratio']
        algorithm = self.optimization_results['algorithm']
        
        investment = float(self.initial_investment.get())
        investment_amounts = [weight * investment for weight in weights]
        
        summary = f"Optimization Results ({algorithm}):\n\n"
        summary += f"Expected Annual Return: {portfolio_return*100:.2f}%\n"
        summary += f"Expected Annual Risk: {portfolio_risk*100:.2f}%\n"
        summary += f"Sharpe Ratio: {sharpe_ratio:.4f}\n\n"
        
        summary += "Portfolio Allocation:\n"
        for i, (stock, weight, amount) in enumerate(zip(stock_names, weights, investment_amounts)):
            if weight > 0.001:  
                summary += f"{stock}: {weight*100:.2f}% (${amount:.2f})\n"
        
        self.results_text.insert(tk.END, summary)
    
    def save_results(self):
        """Save optimization results to a CSV file."""
        if not self.optimization_results:
            messagebox.showwarning("Warning", "No optimization results to save")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            weights = self.optimization_results['weights']
            stock_names = self.optimization_results['stock_names']
            portfolio_return = self.optimization_results['return']
            portfolio_risk = self.optimization_results['risk']
            sharpe_ratio = self.optimization_results['sharpe_ratio']
            algorithm = self.optimization_results['algorithm']
            
            investment = float(self.initial_investment.get())
            results_df = pd.DataFrame({
                'Asset': stock_names,
                'Weight (%)': [w * 100 for w in weights],
                'Amount ($)': [w * investment for w in weights]
            })
            
            summary_df = pd.DataFrame({
                'Metric': ['Algorithm', 'Expected Annual Return (%)', 'Expected Annual Risk (%)', 'Sharpe Ratio'],
                'Value': [algorithm, portfolio_return * 100, portfolio_risk * 100, sharpe_ratio]
            })
            
            with open(filename, 'w', newline='') as f:
                f.write("# Portfolio Optimization Results\n")
                f.write(f"# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("# Summary\n")
                summary_df.to_csv(f, index=False)
                
                f.write("\n# Portfolio Allocation\n")
                results_df.to_csv(f, index=False)
            
            self.status_var.set(f"Results saved to {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
            
    def export_to_pdf(self):
        """Export the current visualization and results to a PDF file."""
        if not self.optimization_results:
            messagebox.showwarning("Warning", "No optimization results to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Save PDF As"
        )
        
        if not filename:
            return
        
        try:
            with PdfPages(filename) as pdf:
                original_size = self.fig.get_size_inches()
                self.fig.set_size_inches(original_size[0]*2.5, original_size[1]*2.5)  
                
                self.fig.tight_layout(pad=4.0)  
                pdf.savefig(self.fig, 
                        bbox_inches='tight', 
                        dpi=300,  
                        )  
                
                self.fig.set_size_inches(original_size)
                
                text_fig = plt.figure(figsize=(10, 12), dpi=100)  
                
                text_fig.text(0.1, 0.95, "Portfolio Optimization Results", 
                            fontsize=16, fontweight='bold')
                
                results_text = self.results_text.get("1.0", tk.END)
                
                lines = results_text.split('\n')
                initial_y = 0.85
                line_spacing = 0.04
                
                for i, line in enumerate(lines):
                    if line.strip():  
                        weight = 'bold' if i == 0 else 'normal'
                        text_fig.text(0.1, initial_y - i*line_spacing, line, 
                                    fontsize=12, fontweight=weight, va='top')
                
              
                text_fig.patch.set_alpha(0)  
                plt.tight_layout(pad=5.0)
                
                pdf.savefig(text_fig, bbox_inches='tight')
                plt.close(text_fig)
                
               
                if hasattr(self, 'additional_figures'):
                    for fig in self.additional_figures:
                        fig.tight_layout()
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                
            self.status_var.set(f"Successfully exported to {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export PDF: {str(e)}")
            if 'text_fig' in locals():
                plt.close(text_fig)

def main():
    root = tk.Tk()
    
    style = ttk.Style()
    style.theme_use('clam')  
    
    style.configure("Accent.TButton", background="#4a7abc", foreground="white")
    style.configure("Green.TButton", background="#4CAF50", foreground="white")
    
    app = PortfolioOptimizationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()