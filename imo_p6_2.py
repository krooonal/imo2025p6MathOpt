from ortools.sat.python import cp_model
from itertools import permutations

def solve_fixed_holes(N):
    for hole_perm in permutations(range(N)):  # hole at (i, hole_perm[i])
        print(f"\nTrying holes at: {[ (i, hole_perm[i]) for i in range(N) ]}")
        if try_tiling_with_fixed_holes(N, hole_perm):
            return  # Stop at first success

def try_tiling_with_fixed_holes(N, hole_perm):
    model = cp_model.CpModel()

    max_tiles = N * N
    x, y, w, h, used = [], [], [], [], []
    x_intervals, y_intervals = [], []

    uncovered = set((i, hole_perm[i]) for i in range(N))
    
    for t in range(max_tiles):
        used_t = model.NewBoolVar(f"used[{t}]")
        used.append(used_t)

        x_t = model.NewIntVar(0, N - 1, f"x[{t}]")
        y_t = model.NewIntVar(0, N - 1, f"y[{t}]")
        x.append(x_t)
        y.append(y_t)

        w_t = model.NewIntVar(1, N, f"w[{t}]")
        h_t = model.NewIntVar(1, N, f"h[{t}]")
        w.append(w_t)
        h.append(h_t)

        model.Add(x_t + w_t <= N).OnlyEnforceIf(used_t)
        model.Add(y_t + h_t <= N).OnlyEnforceIf(used_t)

        x_end = model.NewIntVar(0, N, f"x_end[{t}]")
        y_end = model.NewIntVar(0, N, f"y_end[{t}]")
        model.Add(x_end == x_t + w_t)
        model.Add(y_end == y_t + h_t)

        x_int = model.NewOptionalIntervalVar(x_t, w_t, x_end, used_t, f"x_int[{t}]")
        y_int = model.NewOptionalIntervalVar(y_t, h_t, y_end, used_t, f"y_int[{t}]")
        x_intervals.append(x_int)
        y_intervals.append(y_int)

    model.AddNoOverlap2D(x_intervals, y_intervals)

    # Each non-hole cell must be covered
    for i in range(N):
        for j in range(N):
            # Make sure this cell is covered by some tile
            covering_tiles = []
            for t in range(max_tiles):
                # Check if (i, j) is in tile t's rectangle
                in_tile = model.NewBoolVar(f"covers[{t}][{i},{j}]")

                b1 = model.NewBoolVar("")
                b2 = model.NewBoolVar("")
                b3 = model.NewBoolVar("")
                b4 = model.NewBoolVar("")

                model.Add(x[t] <= j).OnlyEnforceIf(b1)
                model.Add(x[t] > j).OnlyEnforceIf(b1.Not())
                model.Add(j < x[t] + w[t]).OnlyEnforceIf(b2)
                model.Add(j >= x[t] + w[t]).OnlyEnforceIf(b2.Not())

                model.Add(y[t] <= i).OnlyEnforceIf(b3)
                model.Add(y[t] > i).OnlyEnforceIf(b3.Not())
                model.Add(i < y[t] + h[t]).OnlyEnforceIf(b4)
                model.Add(i >= y[t] + h[t]).OnlyEnforceIf(b4.Not())

                model.AddBoolAnd([used[t], b1, b2, b3, b4]).OnlyEnforceIf(in_tile)
                model.AddBoolOr([used[t].Not(), b1.Not(), b2.Not(), b3.Not(), b4.Not()]).OnlyEnforceIf(in_tile.Not())

                covering_tiles.append(in_tile)

            if (i, j) in uncovered:
                model.Add(sum(covering_tiles) == 0)
            else:
                model.AddBoolOr(covering_tiles)

    # Minimize tile count
    model.Minimize(sum(used))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print(f"\nSolution found with {int(solver.ObjectiveValue())} tiles:")
        grid = [['.' for _ in range(N)] for _ in range(N)]
        for t in range(max_tiles):
            if solver.Value(used[t]):
                xt, yt = solver.Value(x[t]), solver.Value(y[t])
                wt, ht = solver.Value(w[t]), solver.Value(h[t])
                for i in range(yt, yt + ht):
                    for j in range(xt, xt + wt):
                        if (i, j) not in uncovered:
                            grid[i][j] = str(t % 10)
        for (i, j) in uncovered:
            grid[i][j] = 'X'
        for row in grid:
            print("".join(row))
        return False
    else:
        print("No solution for this hole configuration.")
        return False

# Try it for N = 4
solve_fixed_holes(4)
