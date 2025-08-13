from ortools.sat.python import cp_model

def solve_min_tiles(N):
    model = cp_model.CpModel()

    max_tiles = N * N  # Conservative upper bound

    x, y, w, h, used = [], [], [], [], []
    x_intervals, y_intervals = [], []

    for t in range(max_tiles):
        used_t = model.NewBoolVar(f"used[{t}]")
        used.append(used_t)

        x_t = model.NewIntVar(0, N - 1, f"x[{t}]")
        y_t = model.NewIntVar(0, N - 1, f"y[{t}]")
        x.append(x_t)
        y.append(y_t)

        w_t = model.NewIntVar(0, N, f"w[{t}]")
        h_t = model.NewIntVar(0, N, f"h[{t}]")
        w.append(w_t)
        h.append(h_t)

        # Enforce dimensions only if used
        model.Add(w_t >= 1).OnlyEnforceIf(used_t)
        model.Add(h_t >= 1).OnlyEnforceIf(used_t)
        model.Add(w_t == 0).OnlyEnforceIf(used_t.Not())
        model.Add(h_t == 0).OnlyEnforceIf(used_t.Not())

        model.Add(x_t + w_t <= N).OnlyEnforceIf(used_t)
        model.Add(y_t + h_t <= N).OnlyEnforceIf(used_t)

        x_end = model.NewIntVar(0, N, f"x_end[{t}]")
        y_end = model.NewIntVar(0, N, f"y_end[{t}]")
        model.Add(x_end == x_t + w_t)
        model.Add(y_end == y_t + h_t)

        x_interval = model.NewOptionalIntervalVar(x_t, w_t, x_end, used_t, f"x_interval[{t}]")
        y_interval = model.NewOptionalIntervalVar(y_t, h_t, y_end, used_t, f"y_interval[{t}]")
        x_intervals.append(x_interval)
        y_intervals.append(y_interval)

    model.AddNoOverlap2D(x_intervals, y_intervals)

    # Coverage indicators
    covered = {}
    for i in range(N):
        for j in range(N):
            covered_ij = model.NewBoolVar(f"covered[{i},{j}]")
            covered[(i, j)] = covered_ij

            tile_covers = []
            for t in range(max_tiles):
                in_x = model.NewBoolVar(f"in_x[{t},{i},{j}]")
                in_y = model.NewBoolVar(f"in_y[{t},{i},{j}]")
                covers = model.NewBoolVar(f"covers[{t},{i},{j}]")

                # j ∈ [x[t], x[t] + w[t])  → x[t] ≤ j < x[t]+w[t]
                b1 = model.NewBoolVar(f"x_le_j[{t},{i},{j}]")
                model.Add(x[t] <= j).OnlyEnforceIf(b1)
                model.Add(x[t] > j).OnlyEnforceIf(b1.Not())

                b2 = model.NewBoolVar(f"j_lt_xw[{t},{i},{j}]")
                model.Add(j < x[t] + w[t]).OnlyEnforceIf(b2)
                model.Add(j >= x[t] + w[t]).OnlyEnforceIf(b2.Not())

                model.AddBoolAnd([b1, b2]).OnlyEnforceIf(in_x)
                model.AddBoolOr([b1.Not(), b2.Not()]).OnlyEnforceIf(in_x.Not())

                # i ∈ [y[t], y[t] + h[t])
                b3 = model.NewBoolVar(f"y_le_i[{t},{i},{j}]")
                model.Add(y[t] <= i).OnlyEnforceIf(b3)
                model.Add(y[t] > i).OnlyEnforceIf(b3.Not())

                b4 = model.NewBoolVar(f"i_lt_yh[{t},{i},{j}]")
                model.Add(i < y[t] + h[t]).OnlyEnforceIf(b4)
                model.Add(i >= y[t] + h[t]).OnlyEnforceIf(b4.Not())

                model.AddBoolAnd([b3, b4]).OnlyEnforceIf(in_y)
                model.AddBoolOr([b3.Not(), b4.Not()]).OnlyEnforceIf(in_y.Not())

                # covers ⇔ used ∧ in_x ∧ in_y
                model.AddBoolAnd([used[t], in_x, in_y]).OnlyEnforceIf(covers)
                model.AddBoolOr([used[t].Not(), in_x.Not(), in_y.Not()]).OnlyEnforceIf(covers.Not())

                tile_covers.append(covers)

            model.Add(sum(tile_covers) == covered_ij)

    # Row & column constraints: each has 1 uncovered square
    for i in range(N):
        model.Add(sum(covered[(i, j)] for j in range(N)) == N - 1)
    for j in range(N):
        model.Add(sum(covered[(i, j)] for i in range(N)) == N - 1)

    model.Minimize(sum(used))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    status = solver.Solve(model)

    # Status report
    print(f"Solver status: {solver.StatusName(status)}")

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print(f"\nSolution found using {int(solver.ObjectiveValue())} tiles on {N}x{N} grid:")
        grid = [['.' for _ in range(N)] for _ in range(N)]
        for t in range(max_tiles):
            if solver.Value(used[t]):
                xt, yt = solver.Value(x[t]), solver.Value(y[t])
                wt, ht = solver.Value(w[t]), solver.Value(h[t])
                print(f"  Tile {t}: top-left=({xt}, {yt}), size={wt}x{ht}")
                for i in range(yt, yt + ht):
                    for j in range(xt, xt + wt):
                        grid[i][j] = str(t % 10)
        print("\nGrid layout (numbers are tile IDs, dots are uncovered):")
        for row in grid:
            print("".join(row))
    elif status == cp_model.INFEASIBLE:
        print("Model is INFEASIBLE — no solution possible with given constraints.")
    elif status == cp_model.MODEL_INVALID:
        print("Model is INVALID — please check constraint definitions.")
    else:
        print("Solver status UNKNOWN — timeout or internal error.")

# Run it for N
solve_min_tiles(N=9)
