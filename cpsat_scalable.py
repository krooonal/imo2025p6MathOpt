from ortools.sat.python import cp_model

def solve_tiling_problem(N, apply_hints=False):
    model = cp_model.CpModel()
    
    hole_cols = [model.NewIntVar(0, N-1, f"hole_col_{i}") for i in range(N)]

    if apply_hints:
        for i in range(N):
            model.add_hint(hole_cols[i], i)
    
    tiles_info = []
    total_area = 0

    for r_start in range(N):
        for c_start in range(N):
            tile_name = f"tile_{r_start}_{c_start}"
            
            is_used = model.NewBoolVar(f"{tile_name}_used")
            width = model.NewIntVar(0, N, f"{tile_name}_width")
            height = model.NewIntVar(0, N, f"{tile_name}_height")
            area = model.NewIntVar(0, N*N, f"{tile_name}_area")
            
            if apply_hints:
                if r_start != c_start:
                    model.add_hint(is_used, True)
                    model.add_hint(width, 1)
                    model.add_hint(height, 1)
                    model.add_hint(area, 1)
                else:
                    model.add_hint(is_used, False)
                    model.add_hint(width, 0)
                    model.add_hint(height, 0)
                    model.add_hint(area, 0)
            
            model.add_multiplication_equality(area, {width, height})

            model.Add(width == 0).OnlyEnforceIf(is_used.Not())
            model.Add(height == 0).OnlyEnforceIf(is_used.Not())
            model.Add(width >= 1).OnlyEnforceIf(is_used)
            model.Add(height >= 1).OnlyEnforceIf(is_used)
            
            model.Add(c_start + width <= N).OnlyEnforceIf(is_used)
            model.Add(r_start + height <= N).OnlyEnforceIf(is_used)

            tiles_info.append({
                "name": tile_name,
                "r_start": r_start,
                "c_start": c_start,
                "width": width,
                "height": height,
                "area": area,
                "used": is_used
            })
            total_area += area
    
    model.AddAllDifferent(hole_cols)
    model.Add(total_area == N*N - N)

    x_intervals = []
    y_intervals = []
    
    for i in range(N):
        hole_x_interval = model.NewIntervalVar(hole_cols[i], 1, hole_cols[i] + 1, f"hole_x_interval_{i}")
        hole_y_interval = model.NewIntervalVar(i, 1, i + 1, f"hole_y_interval_{i}")
        x_intervals.append(hole_x_interval)
        y_intervals.append(hole_y_interval)
        
    for tile in tiles_info:
        x_end = model.NewIntVar(0, N, f"{tile['name']}_x_end")
        y_end = model.NewIntVar(0, N, f"{tile['name']}_y_end")
        model.Add(tile['c_start'] + tile['width'] == x_end)
        model.Add(tile['r_start'] + tile['height'] == y_end)
        x_interval = model.NewOptionalIntervalVar(
            tile['c_start'], tile['width'], x_end, tile['used'], f"x_interval_{tile['name']}")
        y_interval = model.NewOptionalIntervalVar(
            tile['r_start'], tile['height'], y_end, tile['used'], f"y_interval_{tile['name']}")
        x_intervals.append(x_interval)
        y_intervals.append(y_interval)
    
    model.AddNoOverlap2D(x_intervals, y_intervals)

    total_tiles_used = sum(tile['used'] for tile in tiles_info)
    model.Minimize(total_tiles_used)

    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 100
    solver.parameters.num_workers = 8
    status = solver.Solve(model)

    print(f"Solver status: {solver.StatusName(status)}")

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print(f"\nSolution found with {solver.ObjectiveValue()} tiles for N={N}")

        grid = [['.' for _ in range(N)] for _ in range(N)]
        
        for i in range(N):
            c = solver.Value(hole_cols[i])
            grid[i][c] = 'H'
        
        tile_count = 0
        for tile in tiles_info:
            if solver.Value(tile['used']):
                r_start = tile['r_start']
                c_start = tile['c_start']
                width = solver.Value(tile['width'])
                height = solver.Value(tile['height'])
                tile_id = str(tile_count % 10)

                print(f" - Tile {tile_count}: top-left ({r_start}, {c_start}), size={width}x{height}")
                for r in range(r_start, r_start + height):
                    for c in range(c_start, c_start + width):
                        grid[r][c] = tile_id
                tile_count += 1
        
        print("\nGrid:")
        for row in grid:
            print(' '.join(row))
    else:
        print("No solution found.")

solve_tiling_problem(N=4, apply_hints=True)
        
        
        
        
            
            
                    
                

    