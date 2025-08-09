import sys
from ortools.math_opt.python import mathopt
from datetime import timedelta

def solve_tiling_problem(N: int, solver_type_str: str = "HIGHS"):
    if N <= 1:
        print("N must be greater than 1")
        return
    
    print(f"Starting to solve for a {N}x{N} grid using {solver_type_str}...")

    # Create the model
    model = mathopt.Model(name="TilingProblem")

    # variables

    # h[i, j] = 1 if cell (i, j) is a hole.
    h = {}
    for i in range(N):
        for j in range(N):
            h[i, j] = model.add_binary_variable(name=f"h[{i},{j}]")
    
    # (r, c) is the top-left corner of a tile (r_prime, c_prime) is the bottom-right corner of the tile.
    x = {}
    for r in range(N):
        for c in range(N):
            for r_prime in range(N):
                for c_prime in range(N):
                    x[r, c, r_prime, c_prime] = model.add_binary_variable(name=f"x[{r},{c},{r_prime},{c_prime}]")
    
    # objective
    total_tiles = mathopt.fast_sum(x.values())
    model.minimize(total_tiles)

    # constraints
    # Each row must have exactly one hole.
    for i in range(N):
        model.add_linear_constraint(
            mathopt.fast_sum(h[i, j] for j in range(N)) == 1,
            name=f"row_hole_{i}"
        )
    
    # Each column must have exactly one hole.
    for j in range(N):
        model.add_linear_constraint(
            mathopt.fast_sum(h[i, j] for i in range(N)) == 1,
            name=f"col_hole_{j}"
        )
    
    # Covering constraints
    for i in range(N):
        for j in range(N):
            covering_tiles = [
                x[r, c, r_prime, c_prime]
                for r in range(i+1)
                for c in range(j+1)
                for r_prime in range(i, N)
                for c_prime in range(j, N)
            ]
            model.add_linear_constraint(
                h[i, j] + mathopt.fast_sum(covering_tiles) == 1,
                name=f"covering_{i}_{j}"
            )
    
    print("Model built successfully.")

    # solve the model
    try:
        solver_type = mathopt.SolverType[solver_type_str.upper()]
    except KeyError:
        print(f"Invalid solver type: {solver_type_str}")
        solver_type = mathopt.SolverType.HIGHS
    
    solve_params = mathopt.SolveParameters()
    solve_params.time_limit = timedelta(seconds=300)
    solve_params.enable_output = True
    if solver_type != mathopt.SolverType.HIGHS:
        solve_params.threads = 8

    result = mathopt.solve(model, solver_type, params=solve_params)

    if result.termination.reason not in (
        mathopt.TerminationReason.OPTIMAL,
        mathopt.TerminationReason.FEASIBLE,
    ):
        print(f"Solver terminated with reason: {result.termination.reason}")
        return
    
    print("\n" + "=" * 50)
    print("SOLUTION FOUND:")
    print(f"Total tiles used: {int(result.objective_value())}")
    print("Hole locations (row, col):")

    hole_locations = []
    variable_values = result.variable_values()
    for i in range(N):
        for j in range(N):
            if variable_values[h[i, j]] > 0.5:
                hole_locations.append((i, j))
                print(f" - ({i}, {j})")
    print("\nVisual Grid ('H' = Hole, '.' = Covered):")
    grid = [['.' for _ in range(N)] for _ in range(N)]
    for i, j in hole_locations:
        grid[i][j] = 'H'
    for row in grid:
        print(" ".join(row))
    print("=" * 50 + "\n")

if __name__ == "__main__":
    GRID_SIZE = 4
    # Available solvers: "HIGHS", "GSCIP", "GLOP", "CP_SAT"
    SOLVER = "HIGHS"
    solve_tiling_problem(GRID_SIZE, SOLVER)

        
    
    
    
        