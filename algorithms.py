import numpy as np
from collections import defaultdict
import cvxopt as opt
from cvxopt import blas, solvers
import scipy.optimize as sco

def hill_climbing(expected_returns, cov_matrix, iterations=1000, step_size=0.05, 
                 risk_aversion=0.5, allow_short=False, max_weight=1.0, restarts=5):
    """
    Implements Hill Climbing optimization algorithm for portfolio allocation.
    
    Hill Climbing is a local search algorithm that iteratively moves to better solutions
    in the neighborhood of the current solution. This implementation includes multiple
    random restarts to avoid getting stuck in local optima.
    
    Parameters:
    -----------
    expected_returns : numpy.ndarray
        Expected returns for each asset (annualized)
    cov_matrix : numpy.ndarray
        Covariance matrix of asset returns (annualized)
    iterations : int, optional
        Number of iterations per restart (default: 1000)
    step_size : float, optional
        Maximum size of perturbation for generating neighbors (default: 0.05)
    risk_aversion : float, optional
        Risk preference (0 = risk neutral, 1 = risk averse) (default: 0.5)
    allow_short : bool, optional
        Allow negative weights (short positions) (default: False)
    max_weight : float, optional
        Maximum weight for any single asset (default: 1.0)
    restarts : int, optional
        Number of random restarts to perform (default: 5)
    
    Returns:
    --------
    tuple: (optimal_weights, optimization_history)
        optimal_weights : numpy.ndarray
            Optimized portfolio weights
        optimization_history : list
            List of dictionaries containing optimization progress with:
            - weights: Current portfolio weights
            - utility: Current utility score
            - return: Current portfolio return
            - risk: Current portfolio risk
            - iteration: Current iteration count
            - restart: Current restart count
    """
    
    # Initialize variables
    n_assets = len(expected_returns)
    best_weights = None
    best_utility = float('-inf')  # Initialize with worst possible utility
    optimization_history = []
    
    def calculate_utility(weights):
        """
        Calculate portfolio utility using mean-variance framework.
        
        Utility = Portfolio Return - (Risk Aversion * Portfolio Risk)
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Current portfolio weights
            
        Returns:
        --------
        tuple: (utility, portfolio_return, portfolio_risk)
        """
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        utility = portfolio_return - (risk_aversion * portfolio_risk)
        return utility, portfolio_return, portfolio_risk
    
    def generate_random_weights():
        """
        Generate random portfolio weights that satisfy constraints.
        
        Returns:
        --------
        numpy.ndarray
            Random weights normalized to sum to 1 (or sum of absolute values to 1 for short positions)
        """
        if allow_short:
            # weights can be negative but sum of absolute values = 1
            weights = np.random.uniform(-max_weight, max_weight, n_assets)
            weights = weights / np.sum(np.abs(weights))
        else:
            # long position weights between 0 and 1, sum to 1
            weights = np.random.uniform(0, 1, n_assets)
            weights = weights / np.sum(weights)
            
        if not allow_short:
            # ensuring no single asset exceeds the defined max weight
            while np.any(weights > max_weight):
                # calculate total excess weight
                excess = np.sum(np.maximum(0, weights - max_weight))
                # cap weights at max_weight
                weights = np.minimum(weights, max_weight)
                remaining_capacity = np.where(weights < max_weight, max_weight - weights, 0)
                if np.sum(remaining_capacity) > 0:
                    # the exxcess is distributed here across the remaining space
                    weights += remaining_capacity / np.sum(remaining_capacity) * excess
                else:
                    # no capacity remains so normalize
                    weights = weights / np.sum(weights)
        
        return weights
    
    # multiple random restarts to avoid being stuck in local optima
    for restart in range(restarts):
        # initialize current solution with random weights to start
        current_weights = generate_random_weights()
        current_utility, current_return, current_risk = calculate_utility(current_weights)
        
        local_history = [{
            'weights': current_weights.copy(),
            'utility': current_utility,
            'return': current_return,
            'risk': current_risk,
            'iteration': 0,
            'restart': restart
        }]
        
        # hill climbing iterations
        for i in range(1, iterations + 1):
            # checks neighbor solutions
            neighbor_weights = current_weights.copy()
            
            idx1, idx2 = np.random.choice(n_assets, 2, replace=False)
            
            # determine adjustment amount
            adjustment = np.random.uniform(0, step_size)
            
            if allow_short:
                # freely transfer weight between assets if short sell allowed
                neighbor_weights[idx1] += adjustment
                neighbor_weights[idx2] -= adjustment
            else:
                # nsure we dont create negative weights only long positions
                adjustment = min(adjustment, neighbor_weights[idx2])
                neighbor_weights[idx1] += adjustment
                neighbor_weights[idx2] -= adjustment
            
            if not allow_short and neighbor_weights[idx1] > max_weight:
                excess = neighbor_weights[idx1] - max_weight
                neighbor_weights[idx1] = max_weight
                valid_indices = [i for i in range(n_assets) 
                               if i != idx1 and neighbor_weights[i] < max_weight]
                if valid_indices:
                    # Randomly select an asset to receive excess weight
                    idx = np.random.choice(valid_indices)
                    neighbor_weights[idx] += excess
            
            # evaluate neighbor
            neighbor_utility, neighbor_return, neighbor_risk = calculate_utility(neighbor_weights)
            
            # move to neighbor if its better
            if neighbor_utility > current_utility:
                current_weights = neighbor_weights.copy()
                current_utility = neighbor_utility
                current_return = neighbor_return
                current_risk = neighbor_risk
            
            local_history.append({
                'weights': current_weights.copy(),
                'utility': current_utility,
                'return': current_return,
                'risk': current_risk,
                'iteration': i,
                'restart': restart
            })
        
        # update global best if this restart found a better solution
        if current_utility > best_utility:
            best_utility = current_utility
            best_weights = current_weights.copy()
        
        optimization_history.extend(local_history)
    
    return best_weights, optimization_history

def simulated_annealing(expected_returns, cov_matrix, initial_temp=1000, cooling_rate=0.99,
                       iterations=1000, risk_aversion=0.5, allow_short=False,
                       max_weight=1.0, min_temp=1e-3):
    """
    Implements Simulated Annealing optimization algorithm for portfolio allocation.
    
    Simulated Annealing is a probabilistic technique inspired by the annealing process in metallurgy.
    It explores the solution space while gradually decreasing the probability of accepting worse solutions,
    helping to avoid getting stuck in local optima.
    
    Parameters:
    -----------
    expected_returns : numpy.ndarray
        Expected returns for each asset (annualized)
    cov_matrix : numpy.ndarray
        Covariance matrix of asset returns (annualized)
    initial_temp : float, optional
        Initial temperature (controls exploration probability) (default: 1000)
    cooling_rate : float, optional
        Rate at which temperature decreases (0 < cooling_rate < 1) (default: 0.99)
    iterations : int, optional
        Number of iterations at each temperature level (default: 1000)
    risk_aversion : float, optional
        Risk preference (0 = risk neutral, 1 = risk averse) (default: 0.5)
    allow_short : bool, optional
        Allow negative weights (short positions) (default: False)
    max_weight : float, optional
        Maximum weight for any single asset (default: 1.0)
    min_temp : float, optional
        Minimum temperature at which to stop the algorithm (default: 1e-3)
    
    Returns:
    --------
    tuple: (optimal_weights, optimization_history)
        optimal_weights : numpy.ndarray
            Optimized portfolio weights
        optimization_history : list
            List of dictionaries containing optimization progress with:
            - weights: Current portfolio weights
            - utility: Current utility score
            - return: Current portfolio return
            - risk: Current portfolio risk
            - temperature: Current temperature
    """
    
    # Initialize basic parameters
    n_assets = len(expected_returns)
    optimization_history = []

    def calculate_utility(weights):
        """
        Calculate portfolio utility using mean-variance framework.
        
        Utility = Portfolio Return - (Risk Aversion * Portfolio Risk)
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Current portfolio weights
            
        Returns:
        --------
        tuple: (utility, portfolio_return, portfolio_risk)
        """
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        utility = portfolio_return - (risk_aversion * portfolio_risk)
        return utility, portfolio_return, portfolio_risk

    def generate_random_weights():
        """
        Generate random portfolio weights that satisfy constraints.
        
        Returns:
        --------
        numpy.ndarray
            Random weights normalized to sum to 1 (or sum of absolute values to 1 for short positions)
        """
        if allow_short:
            # weights can be negative but their absolute values should sum to 1
            weights = np.random.uniform(-max_weight, max_weight, n_assets)
            weights = weights / np.sum(np.abs(weights))
        else:
            # weights are between 0 and 1 and should sum to 1
            weights = np.random.uniform(0, 1, n_assets)
            weights = weights / np.sum(weights)
        return weights

    # Initialize current and best solutions
    current_weights = generate_random_weights()
    current_utility, current_return, current_risk = calculate_utility(current_weights)
    best_weights = current_weights.copy()
    best_utility = current_utility

    # Start with initial temperature
    temperature = initial_temp

    # Main annealing loop - continues until temperature drops below minimum
    while temperature > min_temp:
        # Perform iterations at current temperature level
        for _ in range(iterations):
            # create a neighbor by slightly adjusting the current weights
            neighbor_weights = current_weights.copy()
            
            # pick two different assets to adjust their weights
            idx1, idx2 = np.random.choice(n_assets, 2, replace=False)
            
            # calculate a random adjustment value
            adjustment = np.random.uniform(0, 0.1)

            if allow_short:
                # for short positions, freely transfer weight between the two assets
                neighbor_weights[idx1] += adjustment
                neighbor_weights[idx2] -= adjustment
            else:
                # for long-only positions, make sure weights don't go negative
                adjustment = min(adjustment, neighbor_weights[idx2])
                neighbor_weights[idx1] += adjustment
                neighbor_weights[idx2] -= adjustment

            # Handle weight constraints
            if not allow_short:
                # adjust weights to stay within [0, max_weight] and normalize them
                neighbor_weights = np.clip(neighbor_weights, 0, max_weight)
                neighbor_weights /= np.sum(neighbor_weights)

            # evaluate the neighbor's solution
            neighbor_utility, neighbor_return, neighbor_risk = calculate_utility(neighbor_weights)

            # Decide whether to accept the neighbor solution
            if neighbor_utility > current_utility or \
               np.random.rand() < np.exp((neighbor_utility - current_utility) / temperature):
                # accept the neighbor solution if it's better or based on probability
                current_weights = neighbor_weights.copy()
                current_utility = neighbor_utility
                current_return = neighbor_return
                current_risk = neighbor_risk

            # update the best solution if the current one is better
            if current_utility > best_utility:
                best_weights = current_weights.copy()
                best_utility = current_utility

            # save the current progress
            optimization_history.append({
                'weights': current_weights.copy(),
                'utility': current_utility,
                'return': current_return,
                'risk': current_risk,
                'temperature': temperature
            })

        # reduce the temperature for the next iteration
        temperature *= cooling_rate

    return best_weights, optimization_history

def genetic_algorithm(expected_returns, cov_matrix, population_size=50, generations=100, 
                      mutation_rate=0.1, crossover_rate=0.8, risk_aversion=0.5, 
                      allow_short=False, max_weight=1.0, elite_count=2):
    """
    Genetic Algorithm for portfolio optimization using roulette wheel selection.

    Parameters:
    - expected_returns (numpy.ndarray): Expected returns for each asset.
    - cov_matrix (numpy.ndarray): Covariance matrix of asset returns.
    - population_size (int): Number of individuals in the population.
    - generations (int): Number of generations to evolve.
    - mutation_rate (float): Probability of mutation.
    - crossover_rate (float): Probability of crossover.
    - risk_aversion (float): Weight given to risk in the utility function.
    - allow_short (bool): Whether short positions are allowed.
    - max_weight (float): Maximum weight for any single asset.
    - elite_count (int): Number of best individuals carried to the next generation.

    Returns:
    - best_weights (numpy.ndarray): Optimized portfolio weights.
    - history (list): Evolution history containing best utility values.
    """

    n_assets = len(expected_returns)
    
    def generate_random_weights():
        """Generate random portfolio weights ensuring valid allocation."""
        weights = np.random.uniform(-max_weight if allow_short else 0, max_weight, n_assets)
        weights /= np.sum(np.abs(weights)) if allow_short else np.sum(weights)
        return weights

    def calculate_fitness(weights):
        """Compute portfolio utility based on expected return and risk."""
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))  # σ²
        fitness = portfolio_return - (risk_aversion * portfolio_variance)  # U(w) = E[R_p] - λ * σ² (objective function)
        return fitness

    def roulette_wheel_selection(population, fitness_scores):
        """Perform roulette wheel selection for parent selection."""
        min_fitness = np.min(fitness_scores)
        adjusted_fitness = fitness_scores - min_fitness + 1e-6  # Shift to positive values
        
        probabilities = adjusted_fitness / np.sum(adjusted_fitness)  # Normalize probabilities based on fitness scores
        selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
        
        return np.array([population[i] for i in selected_indices])  # Select individuals based on probabilities 

    def crossover(parent1, parent2):
        """Perform one-point crossover between two parents."""
        if np.random.rand() < crossover_rate:
            point = np.random.randint(1, n_assets - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return normalize_weights(child1), normalize_weights(child2)
        return parent1, parent2

    def mutate(individual):
        """Apply mutation to introduce diversity in the population."""
        if np.random.rand() < mutation_rate:
            idx = np.random.randint(n_assets)
            individual[idx] += np.random.uniform(-0.1, 0.1)
            individual = normalize_weights(individual)
        return individual

    def normalize_weights(weights):
        """Ensure weights sum up to 1 and stay within valid constraints."""
        weights = np.clip(weights, -max_weight if allow_short else 0, max_weight)
        weights /= np.sum(np.abs(weights)) if allow_short else np.sum(weights)
        return weights

    # Initialize the population with random portfolios
    population = np.array([generate_random_weights() for _ in range(population_size)])
    history = []

    for generation in range(generations):
        # Evaluate fitness of each individual
        fitness_scores = np.array([calculate_fitness(ind) for ind in population])
        
        # Keep the best individuals (elitism)
        elite_indices = np.argsort(fitness_scores)[-elite_count:]
        new_population = [population[i] for i in elite_indices]

        # Select parents using roulette wheel selection
        selected_parents = roulette_wheel_selection(population, fitness_scores)

        # Apply crossover and mutation to generate the new population
        for i in range(0, len(selected_parents) - elite_count, 2):
            child1, child2 = crossover(selected_parents[i], selected_parents[i+1])
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        # Update population for next generation
        population = np.array(new_population[:population_size])

        # Save the best portfolio from this generation
        best_index = np.argmax(fitness_scores)
        history.append({
            'generation': generation,
            'best_utility': fitness_scores[best_index],
            'best_weights': population[best_index].copy()
        })

    # Return the best solution found
    best_solution = max(history, key=lambda x: x['best_utility'])
    return best_solution['best_weights'], history


    
    
def tabu_search(expected_returns, cov_matrix, iterations=1000, step_size=0.05, 
               risk_aversion=0.5, allow_short=False, max_weight=1.0, restarts=5,
               tabu_tenure=20, frequency_memory=True, aspiration_criteria=True):
        """
        Implements tabu search optimization algorithm for portfolio allocation.
        The fundamental concept behind Tabu Search is to restrict or prohibit previously visited moves or states in search areas that don’t offer a better solution.
        The key idea of tabu search is to avoid revisiting the solutions that were previously explored and identified as poor, by maintaining a list of “tabu” solutions.
        We can say Hill climbing + short-term memory = Tabu Search
    
        Parameters:
        expected_returns (numpy.ndarray): Expected returns for each asset (annualized)
        cov_matrix (numpy.ndarray): Covariance matrix of asset returns (annualized)
        iterations (int): Number of iterations for each restart
        step_size (float): Size of step for perturbation
        risk_aversion (float): Weight of risk in the utility function (0-1)
        allow_short (bool): Whether to allow short positions
        max_weight (float): Maximum weight for any single asset
        restarts (int): Number of random restarts
        tabu_tenure (int): Number of iterations a move stays in the tabu list
        frequency_memory (bool): Whether to use frequency memory for diversification
        
        Returns:
        tuple: (optimal_weights, optimization_history)
            - optimal_weights (numpy.ndarray): Optimized portfolio weights
            - optimization_history (list): History of optimization process
        """
        n_assets = len(expected_returns)
        best_weights = None
        best_utility = float('-inf')
        optimization_history = []
        
        def calculate_utility(weights):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            utility = portfolio_return - (risk_aversion * portfolio_risk)
            return utility, portfolio_return, portfolio_risk
        
        def generate_random_weights():
            if allow_short:
                weights = np.random.uniform(-max_weight, max_weight, n_assets)
                weights = weights / np.sum(np.abs(weights))  
            else:
                weights = np.random.uniform(0, 1, n_assets)
                weights = weights / np.sum(weights)  
                
            if not allow_short:
                while np.any(weights > max_weight):
                    excess = np.sum(np.maximum(0, weights - max_weight))
                    weights = np.minimum(weights, max_weight)
                    remaining_capacity = np.where(weights < max_weight, max_weight - weights, 0)
                    if np.sum(remaining_capacity) > 0:
                        weights += remaining_capacity / np.sum(remaining_capacity) * excess
                    else:
                        weights = weights / np.sum(weights) 
            
            return weights
        
        for restart in range(restarts):
            # Initialize tabu list and frequency memory for each restart
            tabu_list = []
            move_frequency = defaultdict(int) if frequency_memory else None
            
            current_weights = generate_random_weights()
            current_utility, current_return, current_risk = calculate_utility(current_weights)
            
            local_history = [{
                'weights': current_weights.copy(),
                'utility': current_utility,
                'return': current_return,
                'risk': current_risk,
                'iteration': 0,
                'restart': restart,
                'tabu_list_size': 0,
                'move_frequency': dict(move_frequency) if frequency_memory else None
            }]
            
            for i in range(1, iterations + 1):
                # Generate neighbors
                neighbors = []
                neighbor_moves = []
                
                # Generate multiple neighbors to choose from
                for _ in range(10):  # Generate 10 potential neighbors
                    neighbor_weights = current_weights.copy()
                    idx1, idx2 = np.random.choice(n_assets, 2, replace=False)
                    
                    adjustment = np.random.uniform(0, step_size)
                    
                    if allow_short:
                        neighbor_weights[idx1] += adjustment
                        neighbor_weights[idx2] -= adjustment
                    else:
                        adjustment = min(adjustment, neighbor_weights[idx2])
                        neighbor_weights[idx1] += adjustment
                        neighbor_weights[idx2] -= adjustment
                    
                    if not allow_short and neighbor_weights[idx1] > max_weight:
                        excess = neighbor_weights[idx1] - max_weight
                        neighbor_weights[idx1] = max_weight
                        valid_indices = [i for i in range(n_assets) if i != idx1 and neighbor_weights[i] < max_weight]
                        if valid_indices:
                            idx = np.random.choice(valid_indices)
                            neighbor_weights[idx] += excess
                    
                    # Create move signature (simplified representation of the move)
                    move_signature = (idx1, idx2, round(adjustment, 4))
                    
                    neighbors.append((neighbor_weights, move_signature))
                    neighbor_moves.append(move_signature)
                
                # Evaluate neighbors and select the best non-tabu move
                best_neighbor_utility = float('-inf')
                best_neighbor_weights = None
                best_move = None
                
                for (neighbor_weights, move) in neighbors:
                    # Check if move is in tabu list
                    is_tabu = move in [t[0] for t in tabu_list]
                    
                    neighbor_utility, _, _ = calculate_utility(neighbor_weights)
                    
                    # Apply frequency penalty if using frequency memory
                    if frequency_memory:
                        freq_penalty = 0.1 * move_frequency[move]  # Small penalty based on frequency
                        neighbor_utility -= freq_penalty
                    
                    # Aspiration criteria - override tabu if better than current best
                    if aspiration_criteria and (neighbor_utility > best_utility):
                        is_tabu = False
                    
                    if not is_tabu and neighbor_utility > best_neighbor_utility:
                        best_neighbor_utility = neighbor_utility
                        best_neighbor_weights = neighbor_weights.copy()
                        best_move = move
                
                # If all moves are tabu, select the least bad one
                if best_neighbor_weights is None:
                    for (neighbor_weights, move) in neighbors:
                        neighbor_utility, _, _ = calculate_utility(neighbor_weights)
                        if neighbor_utility > best_neighbor_utility:
                            best_neighbor_utility = neighbor_utility
                            best_neighbor_weights = neighbor_weights.copy()
                            best_move = move
                
                # Update current solution
                if best_neighbor_weights is not None:
                    current_weights = best_neighbor_weights.copy()
                    current_utility, current_return, current_risk = calculate_utility(current_weights)
                    
                    # Update tabu list
                    if best_move is not None:
                        tabu_list.append((best_move, i + tabu_tenure))  # Add move with expiration iteration
                        if frequency_memory:
                            move_frequency[best_move] += 1
                    
                    # Remove expired tabu moves
                    tabu_list = [move for move in tabu_list if move[1] > i]
                
                # Update best solution if improved
                if current_utility > best_utility:
                    best_utility = current_utility
                    best_weights = current_weights.copy()
                
                # Record history
                local_history.append({
                    'weights': current_weights.copy(),
                    'utility': current_utility,
                    'return': current_return,
                    'risk': current_risk,
                    'iteration': i,
                    'restart': restart,
                    'tabu_list_size': len(tabu_list),
                    'move_frequency': dict(move_frequency) if frequency_memory else None,
                    'current_move': best_move if best_move is not None else None
                })
            
            optimization_history.extend(local_history)
        
        return best_weights, optimization_history

def particle_swarm_optimization(expected_returns, cov_matrix, n_particles=30, iterations=100,
                               inertia_weight=0.7, cognitive_weight=1.5, social_weight=1.5,
                               risk_aversion=0.5, allow_short=False, max_weight=1.0):
    """
    Implements Particle Swarm Optimization (PSO) for portfolio allocation.
    
    PSO is a population-based optimization technique inspired by social behavior of bird flocking.
    Particles (potential solutions) fly through the problem space by following current optimum particles.
    
    Parameters:
    -----------
    expected_returns : numpy.ndarray
        Expected returns for each asset (annualized)
    cov_matrix : numpy.ndarray
        Covariance matrix of asset returns (annualized)
    n_particles : int, optional
        Number of particles in the swarm (default: 30)
    iterations : int, optional
        Maximum number of iterations (default: 100)
    inertia_weight : float, optional
        Controls momentum of particles (default: 0.7)
    cognitive_weight : float, optional
        Weight for particle's personal best (default: 1.5)
    social_weight : float, optional
        Weight for swarm's global best (default: 1.5)
    risk_aversion : float, optional
        Risk preference (0 = risk neutral, 1 = risk averse) (default: 0.5)
    allow_short : bool, optional
        Allow negative weights (short positions) (default: False)
    max_weight : float, optional
        Maximum weight for any single asset (default: 1.0)
    
    Returns:
    --------
    tuple: (optimal_weights, optimization_history)
        optimal_weights : numpy.ndarray
            Optimized portfolio weights
        optimization_history : list
            History of optimization process containing dictionaries with:
            - iteration: Current iteration
            - particle: Particle index
            - weights: Current weights
            - velocity: Current velocity
            - utility: Current utility
            - return: Current return
            - risk: Current risk
            - personal_best_utility: Particle's best utility
            - global_best_utility: Swarm's best utility
    """
    
    # Initialize basic parameters
    n_assets = len(expected_returns)
    optimization_history = []
    
    def calculate_utility(weights):
        """
        Calculate portfolio utility using mean-variance framework.
        
        Utility = Portfolio Return - (Risk Aversion * Portfolio Risk)
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Current portfolio weights
            
        Returns:
        --------
        tuple: (utility, portfolio_return, portfolio_risk)
        """
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        utility = portfolio_return - (risk_aversion * portfolio_risk)
        return utility, portfolio_return, portfolio_risk
    
    def generate_random_weights():
        """
        Generate random portfolio weights that satisfy constraints.
        
        Returns:
        --------
        numpy.ndarray
            Random weights normalized to sum to 1
        """
        if allow_short:
            weights = np.random.uniform(-max_weight, max_weight, n_assets)
            weights = weights / np.sum(np.abs(weights))
        else:
            weights = np.random.uniform(0, 1, n_assets)
            weights = weights / np.sum(weights)
            
        if not allow_short:
            weights = np.clip(weights, 0, max_weight)
            weights = weights / np.sum(weights)
        
        return weights
    
    def normalize_weights(weights):
        """
        Normalize weights to satisfy constraints after velocity updates.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Current unnormalized weights
            
        Returns:
        --------
        numpy.ndarray
            Normalized weights satisfying all constraints
        """
        if allow_short:
            # Normalize for short positions (sum of absolute values = 1)
            if np.sum(np.abs(weights)) > 0:
                weights = weights / np.sum(np.abs(weights))
        else:
            # Ensure no negative weights
            weights = np.maximum(weights, 0)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            
        if not allow_short:
            # Redistribute excess weight if any asset exceeds max_weight
            while np.any(weights > max_weight):
                # Calculate total excess weight
                excess = np.sum(np.maximum(0, weights - max_weight))
                # Cap weights at max_weight
                weights = np.minimum(weights, max_weight)
                # Calculate remaining capacity in other assets
                remaining_capacity = np.where(weights < max_weight, max_weight - weights, 0)
                if np.sum(remaining_capacity) > 0:
                    # Distribute excess proportionally to remaining capacity
                    weights += remaining_capacity / np.sum(remaining_capacity) * excess
                else:
                    # If no capacity remains, just normalize
                    weights = weights / np.sum(weights)
                    
        return weights
    
    # Initialize swarm
    particles = np.array([generate_random_weights() for _ in range(n_particles)])
    velocities = np.array([np.random.uniform(-0.1, 0.1, n_assets) for _ in range(n_particles)])
    
    # Initialize personal bests
    personal_best_positions = particles.copy()
    personal_best_utilities = np.array([calculate_utility(p)[0] for p in particles])
    
    # Initialize global best
    global_best_idx = np.argmax(personal_best_utilities)
    global_best_position = personal_best_positions[global_best_idx].copy()
    global_best_utility, global_best_return, global_best_risk = calculate_utility(global_best_position)
    
    # Record initial state
    for i in range(n_particles):
        utility, ret, risk = calculate_utility(particles[i])
        optimization_history.append({
            'iteration': 0,
            'particle': i,
            'weights': particles[i].copy(),
            'velocity': velocities[i].copy(),
            'utility': utility,
            'return': ret,
            'risk': risk,
            'personal_best_utility': personal_best_utilities[i],
            'global_best_utility': global_best_utility
        })
    
    # main optimization loop
    for t in range(1, iterations + 1):
        # linearly decreasing inertia weight (exploration -> exploitation)
        w = inertia_weight - (inertia_weight - 0.4) * (t / iterations) 
        
        # update each particle
        for i in range(n_particles):
            # generate random factors for stochasticity
            r1 = np.random.random(n_assets)
            r2 = np.random.random(n_assets)
            
            velocities[i] = (w * velocities[i] +  # inertia
                          cognitive_weight * r1 * (personal_best_positions[i] - particles[i]) +  # personal
                          social_weight * r2 * (global_best_position - particles[i])) #global 
            
            # clamp velocity to prevent extreme movements
            velocities[i] = np.clip(velocities[i], -0.2, 0.2)
            
            particles[i] = particles[i] + velocities[i]
            
            # normalize weights to satisfy constraints
            particles[i] = normalize_weights(particles[i])
            
            # calculate new evaluation
            utility, ret, risk = calculate_utility(particles[i])
            
            # update personal best if improved
            if utility > personal_best_utilities[i]:
                personal_best_positions[i] = particles[i].copy()
                personal_best_utilities[i] = utility
                
                # update global best if improved
                if utility > global_best_utility:
                    global_best_position = particles[i].copy()
                    global_best_utility = utility
                    global_best_return = ret
                    global_best_risk = risk
            
            optimization_history.append({
                'iteration': t,
                'particle': i,
                'weights': particles[i].copy(),
                'velocity': velocities[i].copy(),
                'utility': utility,
                'return': ret,
                'risk': risk,
                'personal_best_utility': personal_best_utilities[i],
                'global_best_utility': global_best_utility
            })
    
    return global_best_position, optimization_history

def differential_evolution(expected_returns, cov_matrix, population_size=50, generations=100,
                         F=0.8, CR=0.7, risk_aversion=0.5, allow_short=False, max_weight=1.0):
    """
    Implements Differential Evolution (DE) for portfolio optimization.
    
    DE is a population-based evolutionary algorithm that creates new candidate solutions
    by combining existing ones according to a simple formula, and keeps whichever solution
    has the best fitness score.
    
    Parameters:
    -----------
    expected_returns : numpy.ndarray
        Expected returns for each asset (annualized)
    cov_matrix : numpy.ndarray
        Covariance matrix of asset returns (annualized)
    population_size : int, optional
        Number of individuals in the population (default: 50)
    generations : int, optional
        Number of evolutionary generations (default: 100)
    F : float, optional
        Differential weight/mutation factor (0-2) (default: 0.8)
    CR : float, optional
        Crossover probability (0-1) (default: 0.7)
    risk_aversion : float, optional
        Risk preference (0 = risk neutral, 1 = risk averse) (default: 0.5)
    allow_short : bool, optional
        Allow negative weights (short positions) (default: False)
    max_weight : float, optional
        Maximum weight for any single asset (default: 1.0)
    
    Returns:
    --------
    tuple: (optimal_weights, optimization_history)
        optimal_weights : numpy.ndarray
            Optimized portfolio weights
        optimization_history : list
            List of dictionaries containing optimization progress with:
            - generation: Current generation
            - individual: Individual index
            - weights: Current weights
            - utility: Current utility score
            - return: Current portfolio return
            - risk: Current portfolio risk
            - best_utility: Best utility found so far
    """
    
    # Initialize basic parameters
    n_assets = len(expected_returns)
    optimization_history = []
    
    def calculate_utility(weights):
        """
        Calculate portfolio utility using mean-variance framework.
        
        Utility = Portfolio Return - (Risk Aversion * Portfolio Risk)
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Current portfolio weights
            
        Returns:
        --------
        tuple: (utility, portfolio_return, portfolio_risk)
        """
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        utility = portfolio_return - (risk_aversion * portfolio_risk)
        return utility, portfolio_return, portfolio_risk
    
    def generate_random_weights():
        """
        Generate random portfolio weights that satisfy constraints.
        
        Returns:
        --------
        numpy.ndarray
            Random weights normalized to sum to 1 (or sum of absolute values to 1 for short positions)
        """
        if allow_short:
            weights = np.random.uniform(-max_weight, max_weight, n_assets)
            weights = weights / np.sum(np.abs(weights))
        else:
            weights = np.random.uniform(0, 1, n_assets)
            weights = weights / np.sum(weights)
            
        if not allow_short:
            weights = np.clip(weights, 0, max_weight)
            weights = weights / np.sum(weights)
            
        return weights
    
    def normalize_weights(weights):
        """
        Normalize weights to satisfy constraints after mutation/crossover.
        
        Parameters:
        -----------
        weights : numpy.ndarray
            Current unnormalized weights
            
        Returns:
        --------
        numpy.ndarray
            Normalized weights satisfying all constraints
        """
        if allow_short:
            # normalize for short positions (sum of absolute values = 1)
            if np.sum(np.abs(weights)) > 0:
                weights = weights / np.sum(np.abs(weights))
        else:
            # ensure no negative weights
            weights = np.maximum(weights, 0)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                
        if not allow_short:
            # ensure no weight exceeds maximum allowed
            weights = np.clip(weights, 0, max_weight)
            weights = weights / np.sum(weights)
                
        return weights
    
    # initialize population with random weights
    population = np.array([generate_random_weights() for _ in range(population_size)])
    
    # calculate initial evalutaion for each individual
    fitness = np.array([calculate_utility(ind)[0] for ind in population])
    
    # track best solution found
    best_idx = np.argmax(fitness)
    best_weights = population[best_idx].copy()
    best_utility = fitness[best_idx]
    best_return, best_risk = calculate_utility(best_weights)[1:]

    # record initial population state
    for i in range(population_size):
        utility, ret, risk = calculate_utility(population[i])
        optimization_history.append({
            'generation': 0,
            'individual': i,
            'weights': population[i].copy(),
            'utility': utility,
            'return': ret,
            'risk': risk,
            'best_utility': best_utility
        })
    
    # main evolutionary loop
    for generation in range(1, generations + 1):
        # process each individual in the population
        for i in range(population_size):
            # select three distinct random individuals 
            candidates = [j for j in range(population_size) if j != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            # create mutant vector using differential mutation
            mutant = population[a] + F * (population[b] - population[c])
            
            # ensure mutant satisfies constraints
            mutant = normalize_weights(mutant)
            
            trial = np.zeros(n_assets)
            for j in range(n_assets):
                if np.random.random() < CR or j == np.random.randint(n_assets):
                    trial[j] = mutant[j]
                else:
                    trial[j] = population[i][j]
            
            trial = normalize_weights(trial)
            
            trial_utility, trial_return, trial_risk = calculate_utility(trial)
            
            # selection: replace if better
            if trial_utility > fitness[i]:
                population[i] = trial.copy()
                fitness[i] = trial_utility
                
                # update best solution if improved
                if trial_utility > best_utility:
                    best_weights = trial.copy()
                    best_utility = trial_utility
                    best_return = trial_return
                    best_risk = trial_risk
            
            # record progress for this individual
            optimization_history.append({
                'generation': generation,
                'individual': i,
                'weights': population[i].copy(),
                'utility': fitness[i],
                'return': calculate_utility(population[i])[1],
                'risk': calculate_utility(population[i])[2],
                'best_utility': best_utility
            })
    
    return best_weights, optimization_history

def exact_optimization(expected_returns, cov_matrix, optimization_type='sharpe', 
                      target_return=None, risk_aversion=None,
                      allow_short=False, max_weight=1.0, min_weight=0.0):
    """
    Implements exact optimization algorithms that guarantee optimal solutions for portfolio allocation.
    
    Parameters:
    expected_returns (numpy.ndarray

    tuple: (optimal_weights, results)
        - optimal_weights (numpy.ndarray): Optimized portfolio weights (globally optimal)
        - results (dict): Portfolio statistics including return, risk, etc.
    """
    n_assets = len(expected_returns)
    
    if optimization_type == 'sharpe':
        # maximize sharpe ratio 
        # this is a convex optimization problem that can be solved exactly
        if np.all(expected_returns == expected_returns[0]):
            # all returns equal - minimize variance instead
            optimization_type = 'min_variance'
        else:
            P = opt.matrix(cov_matrix)
            q = opt.matrix(np.zeros(n_assets))
            
            if allow_short:
                G = opt.matrix(np.vstack((-np.eye(n_assets), np.eye(n_assets))))
                h = opt.matrix(np.hstack((-np.ones(n_assets) * min_weight, 
                                         np.ones(n_assets) * max_weight)))
            else:
                G = opt.matrix(np.vstack((-np.eye(n_assets), np.eye(n_assets))))
                h = opt.matrix(np.hstack((np.zeros(n_assets), 
                                         np.ones(n_assets) * max_weight)))
            
            A = opt.matrix(expected_returns.reshape(1, -1))
            b = opt.matrix(1.0)
            
            solvers.options['show_progress'] = False
            solution = solvers.qp(P, q, G, h, A, b)
            
            if solution['status'] != 'optimal':
                raise ValueError("Optimization failed. Status:", solution['status'])
            
            optimal_weights = np.array(solution['x']).flatten()
            optimal_weights /= np.sum(optimal_weights)  # normalize to sum to 1
    
    elif optimization_type == 'min_variance':
        # minimize portfolio variance
        P = opt.matrix(cov_matrix)
        q = opt.matrix(np.zeros(n_assets))
        
        # constraints
        if allow_short:
            G = opt.matrix(np.vstack((-np.eye(n_assets), np.eye(n_assets))))
            h = opt.matrix(np.hstack((-np.ones(n_assets) * min_weight, 
                                     np.ones(n_assets) * max_weight)))
        else:
            G = opt.matrix(np.vstack((-np.eye(n_assets), np.eye(n_assets))))
            h = opt.matrix(np.hstack((np.zeros(n_assets), 
                                     np.ones(n_assets) * max_weight)))
        
        A = opt.matrix(np.ones((1, n_assets)))
        b = opt.matrix(1.0)
        
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        
        if solution['status'] != 'optimal':
            raise ValueError("Optimization failed. Status:", solution['status'])
        
        optimal_weights = np.array(solution['x']).flatten()
    
    elif optimization_type == 'utility':
        # maximize utility: E[r_p] - λ*σ_p^2
        if risk_aversion is None:
            raise ValueError("risk_aversion must be specified for utility optimization")
        
        P = opt.matrix(cov_matrix * risk_aversion * 2) 
        q = opt.matrix(-expected_returns * risk_aversion)
        
        # constraints
        if allow_short:
            G = opt.matrix(np.vstack((-np.eye(n_assets), np.eye(n_assets))))
            h = opt.matrix(np.hstack((-np.ones(n_assets) * min_weight, 
                                     np.ones(n_assets) * max_weight)))
        else:
            G = opt.matrix(np.vstack((-np.eye(n_assets), np.eye(n_assets))))
            h = opt.matrix(np.hstack((np.zeros(n_assets), 
                                     np.ones(n_assets) * max_weight)))
        
        A = opt.matrix(np.ones((1, n_assets)))
        b = opt.matrix(1.0)
        
        
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        
        if solution['status'] != 'optimal':
            raise ValueError("Optimization failed. Status:", solution['status'])
        
        optimal_weights = np.array(solution['x']).flatten()
    
    elif optimization_type == 'target_return':
        # minimize variance for given target return
        if target_return is None:
            raise ValueError("target_return must be specified for target return optimization")
        
        P = opt.matrix(cov_matrix)
        q = opt.matrix(np.zeros(n_assets))
        
        # constraints
        if allow_short:
            G = opt.matrix(np.vstack((-np.eye(n_assets), np.eye(n_assets))))
            h = opt.matrix(np.hstack((-np.ones(n_assets) * min_weight, 
                                     np.ones(n_assets) * max_weight)))
        else:
            G = opt.matrix(np.vstack((-np.eye(n_assets), np.eye(n_assets))))
            h = opt.matrix(np.hstack((np.zeros(n_assets), 
                                     np.ones(n_assets) * max_weight)))
        
        A = opt.matrix(np.vstack((np.ones((1, n_assets)), 
                                expected_returns.reshape(1, -1))))
        b = opt.matrix(np.array([1.0, target_return]))
        
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        
        if solution['status'] != 'optimal':
            raise ValueError("Optimization failed. Status:", solution['status'])
        
        optimal_weights = np.array(solution['x']).flatten()
    
    else:
        raise ValueError("Invalid optimization_type. Choose from 'sharpe', 'min_variance', 'utility', or 'target_return'")
    
    # calculate portfolio statistics
    portfolio_return = np.dot(optimal_weights, expected_returns)
    portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
    
    results = {
        'return': portfolio_return,
        'risk': portfolio_risk,
        'sharpe_ratio': sharpe_ratio,
        'weights': optimal_weights,
        'optimization_type': optimization_type,
        'optimal': True  # flag indicating this is an exact optimal solution
    }
    
    if optimization_type == 'utility':
        results['utility'] = portfolio_return - risk_aversion * (portfolio_risk ** 2)
    
    return optimal_weights, results