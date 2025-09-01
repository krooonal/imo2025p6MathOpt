import collections
import numpy as np
from ortools.linear_solver import pywraplp

NODE_LIMIT = 100
ITERATION_LIMIT = 2000

#Helper function for the pricing problem
def find_max_sum_rectangle(matrix):
    """
    Finds the rectangle in the given matrix with the maximum sum of elements.
    This is the pricing probelm, wich identifies a new tile (column) with the
    most negative reduced cost. It uses an extension of Kadane's algorithm for 2D arrays.
    Args:
        matrix (np.array): The dual matrix where each element (r, c) is the dual
                           value of the constraint for square (r, c).
    Returns:
        tuple: (max_sum, r1, c1, r2, c2) representing the sum of duals for the best
        rectangle and its top-left (r1, c1) and bottom-right (r2, c2) corners.
    """
    rows, cols = matrix.shape
    max_sum = float('-inf')
    best_r1, best_c1, best_r2, best_c2 = -1, -1, -1, -1

    for r_start in range(rows):
        temp_sum = np.zeros(cols)
        for r_end in range(r_start, rows):
            temp_sum += matrix[r_end, :]
        
            # Apply Kadane's algorithm to the 1D array temp_sum
            # This finds the maximum sum subarray within the current column sums
            current_max = 0
            start_col = 0
            for c_end in range(cols):
                current_max += temp_sum[c_end]
                if current_max > max_sum:
                    max_sum = current_max
                    best_r1, best_c1 = r_start, start_col
                    best_r2, best_c2 = r_end, c_end
                if current_max < 0:
                    current_max = 0
                    start_col = c_end + 1
                
    return max_sum, best_r1, best_c1, best_r2, best_c2

class BaseMasterProblem():
    """
    Represents the restricted master problem (RMP) for the tiling problem.
    It can be initialized with branching constraints for a specific node in B&P.
    """
    def __init__(self, N, initial_constraints=None):
        self.solver_ = pywraplp.Solver.CreateSolver("glop")
        self.N_ = N
        self.initial_constraints_ = initial_constraints if initial_constraints is not None else []

        self.generated_tiles_ = set()
        self.tile_vars_ = {}
        self.hole_vars_ = {}
        self.square_constraints_ = {}
        self.row_hole_constraints_ = {}
        self.col_hole_constraints_ = {}
        
        self._build_rmp_structure()
    
    def _build_rmp_structure(self):
        infinity = self.solver_.infinity()
        self.solver_.SetNumThreads(1)

        objective = self.solver_.Objective()
        objective.SetMinimization()

        for r in range(self.N_):
            for c in range(self.N_):
                self.generated_tiles_.add((r, c, r, c))
        
        for r in range(self.N_):
            for c in range(self.N_):
                self.hole_vars_[(r, c)] = self.solver_.NumVar(0.0, 1.0, f"h[{r},{c}]")
        
        for tile in self.generated_tiles_:
            self.tile_vars_[tile] = self.solver_.NumVar(0.0, infinity, f"x[{tile}]")
            objective.SetCoefficient(self.tile_vars_[tile], 1.0)

        for r in range(self.N_):
            for c in range(self.N_):
                square_con = self.solver_.Constraint(1.0, 1.0, f"square[{r},{c}]")
                square_con.SetCoefficient(self.hole_vars_[(r, c)], 1.0)
                for tile in self.generated_tiles_:
                    r1, c1, r2, c2 = tile
                    if r1 <= r <= r2 and c1 <= c <= c2:
                        square_con.SetCoefficient(self.tile_vars_[tile], 1.0)
                self.square_constraints_[(r, c)] = square_con
        
        for i in range(self.N_):
            row_hole_con = self.solver_.Constraint(1.0, 1.0, f"row_hole[{i}]")
            for j in range(self.N_):
                row_hole_con.SetCoefficient(self.hole_vars_[(i, j)], 1.0)
            self.row_hole_constraints_[i] = row_hole_con
        
        for j in range(self.N_):
            col_hole_con = self.solver_.Constraint(1.0, 1.0, f"col_hole[{j}]")
            for i in range(self.N_):
                col_hole_con.SetCoefficient(self.hole_vars_[(i, j)], 1.0)
            self.col_hole_constraints_[j] = col_hole_con
        
        self._apply_branching_constraints()
    
    def _apply_branching_constraints(self):
        for constraint_type, var_id, value in self.initial_constraints_:
            if constraint_type == "hole_fixed":
                r, c = var_id
                hole_var = self.hole_vars_[(r, c)]
                hole_var.SetBounds(value, value)
            elif constraint_type == "tile_fixed":
                tile = var_id
                if tile not in self.tile_vars_:
                    if value == 1:
                        self.add_column(tile[0], tile[1], tile[2], tile[3])
                    else:
                        # if fixed to 0 and not generated, it won't be used.
                        continue
                
                tile_var = self.tile_vars_[tile]
                tile_var.SetBounds(value, value)
    
    def solve_rmp(self):
        solver_params = pywraplp.MPSolverParameters()
        solver_params.SetIntegerParam(
            pywraplp.MPSolverParameters.INCREMENTALITY,
            pywraplp.MPSolverParameters.INCREMENTALITY_ON
        )
        solver_params.SetIntegerParam(
            pywraplp.MPSolverParameters.LP_ALGORITHM,
            pywraplp.MPSolverParameters.PRIMAL
        )
        self.rmp_result_status_ = self.solver_.Solve(solver_params)

        if self.rmp_result_status_ == pywraplp.Solver.OPTIMAL:
            lp_objective_value = self.solver_.Objective().Value()
            dual_matrix = np.zeros((self.N_, self.N_))
            for r in range(self.N_):
                for c in range(self.N_):
                    dual_matrix[r, c] = self.square_constraints_[(r, c)].dual_value()
            return lp_objective_value, dual_matrix
        else:
            return float('inf'), None
    
    def add_column(self, r1, c1, r2, c2):
        """
        Returns false if the column already exists
        """
        new_tile = (r1, c1, r2, c2)
        infinity = self.solver_.infinity()
        if new_tile not in self.generated_tiles_:
            self.generated_tiles_.add(new_tile)
            self.tile_vars_[new_tile] = self.solver_.NumVar(0.0, infinity, f"x[{new_tile}]")
            self.solver_.Objective().SetCoefficient(self.tile_vars_[new_tile], 1)
            
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    self.square_constraints_[(r, c)].SetCoefficient(self.tile_vars_[new_tile], 1)
            return True
        else:
            return False
    
    def get_solution_values(self):
        tile_solution = {tile: var.solution_value() for tile, var in self.tile_vars_.items()}
        hole_solution = {hole: var.solution_value() for hole, var in self.hole_vars_.items()}
        return tile_solution, hole_solution
    
    def solve_final_milp(self):
        """
        Solve a full ILP using all columns that have been generated so far in this node's context.
        This is used to obtain an integer feasible solution for a node if its LP solution is fractional.
        """
        self.ipsolver_ = pywraplp.Solver.CreateSolver("highs")
        self.ipsolver_.SetNumThreads(1)
        
        objective = self.ipsolver_.Objective()
        objective.SetMinimization()
        
        final_tile_vars = {}
        for tile in self.generated_tiles_:
            final_tile_vars[tile] = self.ipsolver_.BoolVar(f"x[{tile}]")
            objective.SetCoefficient(final_tile_vars[tile], 1.0)
        
        final_hole_vars = {}
        for r in range(self.N_):
            for c in range(self.N_):
                final_hole_vars[(r, c)] = self.ipsolver_.BoolVar(f"h[{r},{c}]")
        
        for r in range(self.N_):
            for c in range(self.N_):
                square_con = self.ipsolver_.Constraint(1.0, 1.0, f"square[{r},{c}]")
                square_con.SetCoefficient(final_hole_vars[(r, c)], 1.0)
                for tile in self.generated_tiles_:
                    r1, c1, r2, c2 = tile
                    if r1 <= r <= r2 and c1 <= c <= c2:
                        square_con.SetCoefficient(final_tile_vars[tile], 1.0)
        
        for i in range(self.N_):
            row_hole_con = self.ipsolver_.Constraint(1.0, 1.0, f"row_hole[{i}]")
            for j in range(self.N_):
                row_hole_con.SetCoefficient(final_hole_vars[(i, j)], 1.0)
        
        for j in range(self.N_):
            col_hole_con = self.ipsolver_.Constraint(1.0, 1.0, f"col_hole[{j}]")
            for i in range(self.N_):
                col_hole_con.SetCoefficient(final_hole_vars[(i, j)], 1.0)
        
        # Apply branching constraints
        for constraint_type, var_id, value in self.initial_constraints_:
            if constraint_type == "hole_fixed":
                r, c = var_id
                hole_var = final_hole_vars[(r, c)]
                hole_var.SetBounds(value, value)
            elif constraint_type == "tile_fixed":
                tile = var_id
                if tile in final_tile_vars:
                    tile_var = final_tile_vars[tile]
                    tile_var.SetBounds(value, value)
        
        self.ipsolver_.set_time_limit(100*1000)
        result_status = self.ipsolver_.Solve()
        print(f"ipsolver wall time = {self.ipsolver_.wall_time()}")
        if result_status == pywraplp.Solver.OPTIMAL or result_status == pywraplp.Solver.FEASIBLE:
            return objective.Value(), final_tile_vars, final_hole_vars
        else:
            return float('inf'), None, None

class Node:
    def __init__(self, N, constraints, depth):
        self.N_ = N
        self.constraints_ = constraints
        self.depth_ = depth
        # Each node gets its own master problem to allow independent modifications
        self.master_problem_ = BaseMasterProblem(N, initial_constraints=constraints)
        self.lower_bound_ = 0.0
        self.is_integer_feasible_ = False
        self.tile_solution_ = None
        self.hole_solution_ = None
        self.integer_objective_ = float('inf')
    
    def run_column_generation(self):
        iteration = 0
        while True:
            iteration += 1

            lp_obj_val, dual_matrix = self.master_problem_.solve_rmp()
            self.lower_bound_ = lp_obj_val

            if iteration > ITERATION_LIMIT:
                print("CG iteration limit reached for this node.")
                break

            if dual_matrix is None:
                self.lower_bound_ = float('inf')
                break
            
            # Pricing problem: Find a tile with the most negative reduced cost
            max_dual_sum, r1, c1, r2, c2 = find_max_sum_rectangle(dual_matrix)
            reduced_cost = 1.0 - max_dual_sum

            if reduced_cost < -1e-6:
                self.master_problem_.add_column(r1, c1, r2, c2)
            else:
                break
        
        # Check if the LP solution is integer feasible
        tile_sol_lp, hole_sol_lp = self.master_problem_.get_solution_values()
        is_integer_feasible = tile_sol_lp is not None
        if tile_sol_lp is not None:
            for val in tile_sol_lp.values():
                if abs(val - round(val)) > 1e-6:
                    is_integer_feasible = False
                    break
            if is_integer_feasible:
                for val in hole_sol_lp.values():
                    if abs(val - round(val)) > 1e-6:
                        is_integer_feasible = False
                        break
         
        self.is_integer_feasible_ = is_integer_feasible
        if self.is_integer_feasible_:
            self.integer_objective_ = self.lower_bound_
            self.tile_solution_ = {t: round(v) for t, v in tile_sol_lp.items()}
            self.hole_solution_ = {h: round(v) for h, v in hole_sol_lp.items()}
        else:
            if self.depth_ >= 1:
                return
            milp_obj, milp_tile_vars, milp_hole_vars = self.master_problem_.solve_final_milp()
            if milp_obj < float('inf'):
                self.integer_objective_ = milp_obj
                self.tile_solution_ = {t: round(v.solution_value()) for t, v in milp_tile_vars.items()}
                self.hole_solution_ = {h: round(v.solution_value()) for h, v in milp_hole_vars.items()}

class BranchAndPriceSolver:
    def __init__(self, N):
        self.N_ = N
        self.nodes_ = collections.deque()
        self.upper_bound_ = float('inf')
        self.best_integer_solution_ = None
        self.best_hole_solution_ = None
        self.nodes_processed_ = 0
    
    def solve(self):
        root_node = Node(self.N_, constraints=[], depth=0)
        self.nodes_.append(root_node)
        print(f"Starting Branch and Price for {self.N_}x{self.N_} grid.")

        while self.nodes_:
            self.nodes_processed_ += 1
            current_node = None
            if self.nodes_processed_ >= 2:
                current_node = self.nodes_.pop()
            else:
                current_node = self.nodes_.popleft()
            
            if self.nodes_processed_ >= NODE_LIMIT:
                print("Node limit reached. Ending Branch and Price.")
                break
                
            print(f"Processing node (Depth: {current_node.depth_}, Processed: {self.nodes_processed_}) ---")
            print(f"Current best upper bound: {self.upper_bound_:.4f}")

            if current_node.lower_bound_ >= self.upper_bound_:
                print(f"Pruning node (lower bound {current_node.lower_bound_:.4f} >= current upper bound {self.upper_bound_:.4f})")
                continue

            current_node.run_column_generation()
            print(f"Node LP Lower Bound after CG: {current_node.lower_bound_:.4f}")

            # Recheck after CG as the lower bound might have changed
            if current_node.lower_bound_ >= self.upper_bound_:
                print(f"Pruning node after CG (lower bound {current_node.lower_bound_:.4f} >= current upper bound {self.upper_bound_:.4f})")
                continue

            if current_node.is_integer_feasible_:
                print(f"Node found integer solution with objective {current_node.integer_objective_:.4f}")
                if current_node.integer_objective_ < self.upper_bound_:
                    self.upper_bound_ = current_node.integer_objective_
                    self.best_integer_solution_ = current_node.tile_solution_
                    self.best_hole_solution_ = current_node.hole_solution_
                    print(f"*** Updated global upper bound to {self.upper_bound_:.4f} ***")
                if current_node.depth_ > 0:
                    continue
            
            fractional_var_found = False
            # Prioritize branching on hole variables
            for (r, c), var in current_node.master_problem_.hole_vars_.items():
                val = var.solution_value()
                if abs(val - round(val)) > 1e-6:
                    print(f"Branching on hole variable h[{r},{c}] = {val:.4f}")

                    new_constraints_0 = current_node.constraints_ + [("hole_fixed", (r, c), 0)]
                    child_node_0 = Node(self.N_, new_constraints_0, current_node.depth_ + 1)
                    self.nodes_.append(child_node_0)
                    
                    new_constraints_1 = current_node.constraints_ + [("hole_fixed", (r, c), 1)]
                    child_node_1 = Node(self.N_, new_constraints_1, current_node.depth_ + 1)
                    self.nodes_.append(child_node_1)
                    fractional_var_found = True
                    break
            
            if not fractional_var_found:
                for tile, var in current_node.master_problem_.tile_vars_.items():
                    val = var.solution_value()
                    if abs(val - round(val)) > 1e-6:
                        print(f"Branching on tile variable x[{tile}] = {val:.4f}")

                        new_constraints_0 = current_node.constraints_ + [("tile_fixed", tile, 0)]
                        child_node_0 = Node(self.N_, new_constraints_0, current_node.depth_ + 1)
                        self.nodes_.append(child_node_0)
                        
                        new_constraints_1 = current_node.constraints_ + [("tile_fixed", tile, 1)]
                        child_node_1 = Node(self.N_, new_constraints_1, current_node.depth_ + 1)
                        self.nodes_.append(child_node_1)
                        fractional_var_found = True
                        break
            
            if not fractional_var_found:
                print("Warning: No fractional variables found in node but node LP solution is not integer feasible.")
                if current_node.integer_objective_ < self.upper_bound_:
                    self.upper_bound_ = current_node.integer_objective_
                    self.best_integer_solution_ = current_node.tile_solution_
                    self.best_hole_solution_ = current_node.hole_solution_
                    print(f"*** Updated global upper bound to {self.upper_bound_:.4f} ***")
            
        print("\n--- Branch and Price Finished ---")
        print(f"Total nodes processed: {self.nodes_processed_}")
        if self.best_integer_solution_ is not None:
            print(f"Best Objective Value: {self.upper_bound_:.0f}")
            print("Selected Tiles:")
            for tile, val in self.best_integer_solution_.items():
                if val > 0.5:
                    print(f"    Tile from ({tile[0]},{tile[1]}) to ({tile[2]},{tile[3]})")
            
            print("Hole Positions:")
            hole_grid = [['.' for _ in range(self.N_)] for _ in range(self.N_)]
            for (r, c), val in self.best_hole_solution_.items():
                if val > 0.5:
                    print(f"    Hole at ({r},{c})")
                    hole_grid[r][c] = 'H'
            
            print("Resulting grid (H = Hole, . = Covered)")
            for row in hole_grid:
                print(" ".join(row))
            
        else:
            print("No integer solution found. The search was incomplete.")

# Solve
solver = BranchAndPriceSolver(N = 15)
solver.solve()

                    
        
        
            
            
        
            
            
            
        
        
        
        
                
            
            
            
                
                
        
                
            
            
    