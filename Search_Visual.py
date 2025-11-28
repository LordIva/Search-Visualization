import time
import threading
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from typing import List, Set, Optional
from collections import deque

# =========================
# 1) Core dataclasses
# =========================

@dataclass(frozen=True)
class Position:
    x: int
    y: int

@dataclass(frozen=True)
class GameState:
    position: Position
    energy: int
    visited_foods: frozenset  # use frozenset for immutability/hashability


# =========================
# 2) Game rules/engine
# =========================

class TreasureHuntGame:
    TERRAIN_COSTS = {
        '.': 1,        # Normal
        '~': 2,        # Swamp
        '^': 3,        # Hills
        'S': 1,        # Start
        'T': 1,        # Treasure
        'F': 1,        # Food
        'X': float('inf')  # Obstacle
    }

    def __init__(self, game_map: List[str], starting_energy: int = 12, max_energy: int = 20, food_energy: int = 5):
        self.game_map = [list(row) for row in game_map]
        self.height = len(self.game_map)
        self.width = len(self.game_map[0]) if self.height else 0
        self.starting_energy = starting_energy
        self.max_energy = max_energy
        self.food_energy = food_energy

        self.start_pos = self._find_position('S')
        self.treasure_pos = self._find_position('T')
        self.food_positions = set(self._find_all_positions('F'))

    def _find_position(self, symbol: str) -> Position:
        for y in range(self.height):
            for x in range(self.width):
                if self.game_map[y][x] == symbol:
                    return Position(x, y)
        raise ValueError(f"Symbol '{symbol}' not found on map")

    def _find_all_positions(self, symbol: str) -> List[Position]:
        out = []
        for y in range(self.height):
            for x in range(self.width):
                if self.game_map[y][x] == symbol:
                    out.append(Position(x, y))
        return out

    def get_initial_state(self) -> GameState:
        return GameState(
            position=self.start_pos,
            energy=self.starting_energy,
            visited_foods=frozenset()
        )

    def is_valid_position(self, pos: Position) -> bool:
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height

    def get_terrain_at(self, pos: Position) -> str:
        if not self.is_valid_position(pos):
            return 'X'
        return self.game_map[pos.y][pos.x]

    def get_possible_moves(self, state: GameState) -> List[GameState]:
        moves = []
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        for dx, dy in directions:
            new_pos = Position(state.position.x + dx, state.position.y + dy)
            if not self.is_valid_position(new_pos):
                continue

            terrain = self.get_terrain_at(new_pos)
            if terrain == 'X':
                continue

            energy_cost = self.TERRAIN_COSTS.get(terrain, 1)
            new_energy = state.energy - energy_cost
            if new_energy <= 0:
                continue

            visited_foods = set(state.visited_foods)
            if new_pos in self.food_positions and new_pos not in visited_foods:
                new_energy = min(new_energy + self.food_energy, self.max_energy)
                visited_foods.add(new_pos)

            moves.append(GameState(new_pos, new_energy, frozenset(visited_foods)))

        return moves

    def is_goal_state(self, state: GameState) -> bool:
        return state.position == self.treasure_pos
    def heuristic(self, pos: Position) -> float:
        """Manhattan distance heuristic to treasure"""
        return abs(pos.x - self.treasure_pos.x) + abs(pos.y - self.treasure_pos.y)


# =========================
# 3) Blind search algorithms
# =========================

class SearchAlgorithms:
    def __init__(self, game: TreasureHuntGame):
        self.game = game

    def bfs(self):
        start = self.game.get_initial_state()
        if self.game.is_goal_state(start):
            return True, [start]

        q = deque([(start, [start])])
        visited = {start}

        while q:
            state, path = q.popleft()
            for nxt in self.game.get_possible_moves(state):
                if nxt in visited:
                    continue
                new_path = path + [nxt]
                if self.game.is_goal_state(nxt):
                    return True, new_path
                q.append((nxt, new_path))
                visited.add(nxt)

        return False, []

    def dfs(self, max_depth: int = 100):
        start = self.game.get_initial_state()
        if self.game.is_goal_state(start):
            return True, [start]

        stack = [(start, [start], 0)]
        visited = set()

        while stack:
            state, path, depth = stack.pop()
            if state in visited or depth >= max_depth:
                continue
            visited.add(state)

            if self.game.is_goal_state(state):
                return True, path

            for nxt in self.game.get_possible_moves(state):
                if nxt not in visited:
                    stack.append((nxt, path + [nxt], depth + 1))

        return False, []

    def ids(self, max_depth: int = 50):
        for depth in range(max_depth + 1):
            ok, path = self._dls(depth)
            if ok:
                return True, path
        return False, []

    def _dls(self, depth_limit: int):
        start = self.game.get_initial_state()
        stack = [(start, [start], 0)]
        visited_at_depth = set()

        while stack:
            state, path, depth = stack.pop()
            if depth > depth_limit:
                continue
            key = (state, depth)
            if key in visited_at_depth:
                continue
            visited_at_depth.add(key)

            if self.game.is_goal_state(state):
                return True, path

            if depth < depth_limit:
                for nxt in self.game.get_possible_moves(state):
                    stack.append((nxt, path + [nxt], depth + 1))
        return False, []


# =========================
# 4) Sample maps
# =========================

test_maps = {
    "original": [
        "S.~F.X^^^",
        ".X~~.X^F^",
        ".X.F~~.X.",
        "F..X~..^.",
        "~~X..F^^T"
    ],
    "maze_like": [
        "S.X.F.X..",
        ".XXXXX.X.",
        ".......X.",
        "XXXX.XXX.",
        "F..X...X.",
        ".X.XXX.X.",
        ".......X.",
        "XXXXX.X..",
        "F.....XFT"
    ],
    "food_desert": [
        "S........",
        ".........",
        ".........",
        "....F....",
        ".........",
        ".........",
        ".........",
        ".........",
        "........T"
    ],
    "swamp_challenge": [
        "S.......F",
        "~~~~~~~~~",
        "~~~~~~~~~",
        "~~~~~~~~~",
        "F~~~F~~~F",
        "~~~~~~~~~",
        "~~~~~~~~~",
        "~~~~~~~~~",
        "F.......T"
    ],
    "dfs_trap": [
        "S.........",
            "XXXXXXXXX.",
            "........X.",
            ".XXXXXXXX.",
            "..........",
            "XXXXXXXXX.",
            "X.........",
            ".XXXXXXXX.",
            "F.......T."
        ],
    "energy_vs_path" : [
    "S..^.",
    ".^.^.",
    "...~.",
    ".^^^.",
    "..F.T"
],
    "energy_vs_path2" : [
    
    "S.~~T",
    "XX...",
    ".F..."


]
}


# =========================
# 5) Visualization
# =========================

@dataclass
class VisualizationStep:
    current_state: GameState
    visited_states: Set[GameState]
    frontier_states: Set[GameState]
    step_type: str   
    algorithm: str
    step_number: int

class TreasureHuntVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Treasure Hunt Algorithm Visualizer")
        self.root.geometry("1200x800")

        self.colors = {
            'S': '#00FF00',
            'T': '#FFD700',
            'F': '#FFA500',
            'X': '#8B4513',
            '.': '#FFFFFF',
            '~': '#87CEEB',
            '^': '#A0522D',
            'visited': '#FFCCCC',
            'frontier': '#CCFFCC',
            'current': '#FF0000',
            'path': '#FF00FF'
        }

        self.game: Optional[TreasureHuntGame] = None
        self.search_algorithms: Optional[SearchAlgorithms] = None
        self.visualization_steps: List[VisualizationStep] = []
        self.current_step = 0
        self.is_playing = False
        self.play_speed = 500
        self.cell_size = 40

        self._build_ui()
        self._load_map("original")

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        controls = ttk.LabelFrame(main, text="Controls", padding=10)
        controls.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(controls, text="Algorithm:").grid(row=0, column=0, padx=5)
        self.algorithm_var = tk.StringVar(value="BFS")
        algo = ttk.Combobox(controls, textvariable=self.algorithm_var,
                            values=["BFS", "DFS", "IDS", "Greedy", "A* Energy-Optimal","A* Path-Optimal"], state="readonly")
        algo.grid(row=0, column=1, padx=5)

        ttk.Label(controls, text="Map:").grid(row=0, column=2, padx=5)
        self.map_var = tk.StringVar(value="original")
        self.map_combo = ttk.Combobox(controls, textvariable=self.map_var,
                                      values=list(test_maps.keys()), state="readonly")
        self.map_combo.grid(row=0, column=3, padx=5)
        self.map_combo.bind("<<ComboboxSelected>>", self._on_map_change)

        ttk.Label(controls, text="Speed (ms):").grid(row=0, column=4, padx=5)
        self.speed_var = tk.IntVar(value=500)
        speed = ttk.Scale(controls, from_=100, to=2000, variable=self.speed_var, orient=tk.HORIZONTAL)
        speed.grid(row=0, column=5, padx=5)

        btns = ttk.Frame(controls)
        btns.grid(row=1, column=0, columnspan=6, pady=10)

        self.run_btn = ttk.Button(btns, text="Run Algorithm", command=self._run_algorithm_thread)
        self.run_btn.pack(side=tk.LEFT, padx=5)

        self.play_btn = ttk.Button(btns, text="Play", command=self._play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.pause_btn = ttk.Button(btns, text="Pause", command=self._pause, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        self.step_btn = ttk.Button(btns, text="Step", command=self._step, state=tk.DISABLED)
        self.step_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = ttk.Button(btns, text="Reset", command=self._reset, state=tk.DISABLED)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        content = ttk.Frame(main)
        content.pack(fill=tk.BOTH, expand=True)

        canvas_frame = ttk.LabelFrame(content, text="Game Map", padding=10)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.canvas = tk.Canvas(canvas_frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        info_frame = ttk.LabelFrame(content, text="Information", padding=10)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y)
        info_frame.configure(width=320)

        self.step_info = tk.Text(info_frame, height=12, width=38, wrap=tk.WORD)
        self.step_info.pack(fill=tk.X, pady=(0, 10))

        self.stats_info = tk.Text(info_frame, height=20, width=38, wrap=tk.WORD)
        self.stats_info.pack(fill=tk.BOTH, expand=True)

        progress_frame = ttk.Frame(main)
        progress_frame.pack(fill=tk.X, pady=10)

        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=1)
        self.progress_bar.pack(fill=tk.X)

        self.progress_label = ttk.Label(progress_frame, text="Step 0 of 0")
        self.progress_label.pack()

        # Redraw map when canvas resizes
        self.canvas.bind("<Configure>", lambda e: self._draw_map())

    # ---------- Map / drawing ----------

    def _load_map(self, name: str):
        grid = test_maps[name]
        self.game = TreasureHuntGame(grid)
        self.search_algorithms = SearchAlgorithms(self.game)
        self.visualization_steps.clear()
        self.current_step = 0
        self._draw_map()
        self._update_map_info()
        self._set_controls_ready(False)

    def _on_map_change(self, _evt=None):
        self._pause()
        self._load_map(self.map_var.get())

    def _draw_map(self):
        if not self.game:
            return
        self.canvas.delete("all")
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 600
        cell_w = cw // self.game.width
        cell_h = ch // self.game.height
        self.cell_size = max(16, min(cell_w, cell_h, 48))

        for y in range(self.game.height):
            for x in range(self.game.width):
                x1 = x * self.cell_size
                y1 = y * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                terrain = self.game.game_map[y][x]
                color = self.colors.get(terrain, '#FFFFFF')
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='black')
                self.canvas.create_text(x1 + self.cell_size // 2, y1 + self.cell_size // 2,
                                        text=terrain, font=('Arial', 8, 'bold'))

    def _highlight_cell(self, x, y, color, tag):
        x1 = x * self.cell_size + 2
        y1 = y * self.cell_size + 2
        x2 = x1 + self.cell_size - 4
        y2 = y1 + self.cell_size - 4
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='', stipple='gray50', tags=tag)

    # ---------- Run & collect steps (with thread) ----------

    def _run_algorithm_thread(self):
        if not self.game:
            return
        self._pause()
        self._set_controls_busy(True)
        self.visualization_steps.clear()
        self.current_step = 0
        self._draw_map()
        self._update_progress()

        t = threading.Thread(target=self._run_algorithm_collect_steps, daemon=True)
        t.start()

    def _run_algorithm_collect_steps(self):
        start_time = time.time()
        algo = self.algorithm_var.get()
        if algo == "BFS":
            result = self._run_bfs_with_steps()
        elif algo == "DFS":
            result = self._run_dfs_with_steps()
        elif algo == "IDS":
            result = self._run_ids_with_steps()
        elif algo == "Greedy":
            result = self._run_greedy_with_steps()
        elif algo == "A* Energy-Optimal":
            result = self._run_astar_with_steps()
        elif algo =="A* Path-Optimal":
            result=self.run_astar_path_optimal()
        else:
            result = {'success': False, 'path': [], 'steps': 0, 'algorithm': algo}


        elapsed = (time.time() - start_time) * 1000.0
        # Back to UI thread
        self.root.after(0, lambda: self._after_run(result, elapsed))

    def _after_run(self, result, elapsed_ms: float):
        self.progress_bar.config(maximum=max(1, len(self.visualization_steps)))
        self._update_progress()
        self._update_statistics(result, elapsed_ms)
        self._set_controls_ready(True)

        # Start path animation if success
        if result['success'] and 'path' in result:
            self._animate_path(result['path'], 0)
        self.final_path = result['path'] if result['success'] else []

 
    def _animate_path(self, path: list, index: int):
        """Recursively animate the final path"""
        if index >= len(path):
            return

        state = path[index]
        x, y = state.position.x, state.position.y
        self._highlight_cell(x, y, self.colors['path'], tag="path")

        # Delay between highlighting each cell
        self.root.after(200, lambda: self._animate_path(path, index + 1))

    # ---------- Step builders for visualization ----------

    def _run_bfs_with_steps(self):
        initial = self.game.get_initial_state()
        q = deque([(initial, [initial])])
        visited = {initial}
        step_no = 0

        while q:
            current, path = q.popleft()
            frontier = {st for st, _ in q}
            self.visualization_steps.append(
                VisualizationStep(current, visited.copy(), frontier, 'exploring', 'BFS', step_no)
            )
            step_no += 1

            if self.game.is_goal_state(current):
                self.visualization_steps.append(
                    VisualizationStep(current, visited.copy(), set(), 'found_goal', 'BFS', step_no)
                )
                return {'success': True, 'path': path, 'steps': step_no, 'algorithm': 'BFS'}

            for nxt in self.game.get_possible_moves(current):
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, path + [nxt]))

        self.visualization_steps.append(
            VisualizationStep(current, visited.copy(), set(), 'failed', 'BFS', step_no)
        )
        return {'success': False, 'path': [], 'steps': step_no, 'algorithm': 'BFS'}

    def _run_dfs_with_steps(self, max_depth: int = 100):
        initial = self.game.get_initial_state()
        stack = [(initial, [initial], 0)]
        visited = set()
        step_no = 0

        while stack:
            current, path, depth = stack.pop()
            if current in visited or depth >= max_depth:
                continue
            visited.add(current)

            frontier = {st for st, _, _ in stack}
            self.visualization_steps.append(
                VisualizationStep(current, visited.copy(), frontier, 'exploring', 'DFS', step_no)
            )
            step_no += 1

            if self.game.is_goal_state(current):
                self.visualization_steps.append(
                    VisualizationStep(current, visited.copy(), set(), 'found_goal', 'DFS', step_no)
                )
                return {'success': True, 'path': path, 'steps': step_no, 'algorithm': 'DFS'}

            for nxt in self.game.get_possible_moves(current):
                if nxt not in visited:
                    stack.append((nxt, path + [nxt], depth + 1))

        last = current if 'current' in locals() else initial
        self.visualization_steps.append(
            VisualizationStep(last, visited.copy(), set(), 'failed', 'DFS', step_no)
        )
        return {'success': False, 'path': [], 'steps': step_no, 'algorithm': 'DFS'}

    def _run_ids_with_steps(self, max_depth: int = 30):
        # Build steps by running DLS multiple times, tagging algorithm as IDS
        initial = self.game.get_initial_state()
        total_steps = 0
        for limit in range(max_depth + 1):
            stack = [(initial, [initial], 0)]
            visited_at_depth = set()

            while stack:
                current, path, depth = stack.pop()
                if depth > limit:
                    continue
                key = (current, depth)
                if key in visited_at_depth:
                    continue
                visited_at_depth.add(key)

                # For visualization, derive visited/frontier sets
                frontier_states = {st for st, _, _ in stack}
                visited_states = {s for (s, d) in visited_at_depth if d <= depth}

                self.visualization_steps.append(
                    VisualizationStep(current, visited_states.copy(), frontier_states.copy(),
                                      'exploring', 'IDS', total_steps)
                )
                total_steps += 1

                if self.game.is_goal_state(current):
                    self.visualization_steps.append(
                        VisualizationStep(current, visited_states.copy(), set(),
                                          'found_goal', 'IDS', total_steps)
                    )
                    return {'success': True, 'path': path, 'steps': total_steps, 'algorithm': 'IDS'}

                if depth < limit:
                    for nxt in self.game.get_possible_moves(current):
                        stack.append((nxt, path + [nxt], depth + 1))

        # Failed overall
        self.visualization_steps.append(
            VisualizationStep(initial, set(), set(), 'failed', 'IDS', total_steps)
        )
        return {'success': False, 'path': [], 'steps': total_steps, 'algorithm': 'IDS'}
        # ---------- Greedy Best-First Search (GBFS) ----------
    
    def _run_greedy_with_steps(self):
        import heapq, itertools
        counter = itertools.count()
        initial = self.game.get_initial_state()
        pq = [(self.game.heuristic(initial.position), next(counter), initial, [initial])]
        visited = {initial}
        step_no = 0

        while pq:
            _, _, current, path = heapq.heappop(pq)
            frontier = {st for _, _, st, _ in pq}
            self.visualization_steps.append(
                VisualizationStep(current, visited.copy(), frontier, 'exploring', 'Greedy', step_no)
            )
            step_no += 1

            if self.game.is_goal_state(current):
                self.visualization_steps.append(
                    VisualizationStep(current, visited.copy(), set(), 'found_goal', 'Greedy', step_no)
                )
                return {'success': True, 'path': path, 'steps': step_no, 'algorithm': 'Greedy'}

            for nxt in self.game.get_possible_moves(current):
                if nxt not in visited:
                    visited.add(nxt)
                    h = self.game.heuristic(nxt.position)
                    heapq.heappush(pq, (h, next(counter), nxt, path + [nxt]))

        self.visualization_steps.append(
            VisualizationStep(current, visited.copy(), set(), 'failed', 'Greedy', step_no)
        )
        return {'success': False, 'path': [], 'steps': step_no, 'algorithm': 'Greedy'}    
        return {'success': False, 'path': [], 'steps': step_no, 'algorithm': 'Greedy'}

    # ---------- A* Search ----------
    def _run_astar_with_steps(self):
        import heapq, itertools
        counter = itertools.count()
        initial = self.game.get_initial_state()
        g_scores = {initial: 0}
        pq = [(self.game.heuristic(initial.position), next(counter), 0, initial, [initial])]
        visited = set()
        step_no = 0

        while pq:
            f, _, g, current, path = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)

            frontier = {st for _, _, _, st, _ in pq}
            self.visualization_steps.append(
                VisualizationStep(current, visited.copy(), frontier, 'exploring', 'A*', step_no)
            )
            step_no += 1

            if self.game.is_goal_state(current):
                self.visualization_steps.append(
                    VisualizationStep(current, visited.copy(), set(), 'found_goal', 'A*', step_no)
                )
                return {'success': True, 'path': path, 'steps': step_no, 'algorithm': 'A*'}

            for nxt in self.game.get_possible_moves(current):
                cost = self.game.TERRAIN_COSTS[self.game.get_terrain_at(nxt.position)]
                new_g = g + cost
                if nxt not in g_scores or new_g < g_scores[nxt]:
                    g_scores[nxt] = new_g
                    f_score = new_g + self.game.heuristic(nxt.position)
                    heapq.heappush(pq, (f_score, next(counter), new_g, nxt, path + [nxt]))

        self.visualization_steps.append(
            VisualizationStep(current, visited.copy(), set(), 'failed', 'A*', step_no)
        )
        return {'success': False, 'path': [], 'steps': step_no, 'algorithm': 'A*'}

    def run_astar_path_optimal(self):
    
        import heapq, itertools
        counter = itertools.count()
        initial = self.game.get_initial_state()
        g_scores = {initial: 0}
        pq = [(self.game.heuristic(initial.position), next(counter), 0, initial, [initial])]
        visited = set()
        step_no = 0

        while pq:
            f, _, g, current, path = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)

            frontier = {st for _, _, _, st, _ in pq}
            self.visualization_steps.append(
                VisualizationStep(current, visited.copy(), frontier, 'exploring', 'A* Path-Optimal', step_no)
            )
            step_no += 1

            if self.game.is_goal_state(current):
                self.visualization_steps.append(
                    VisualizationStep(current, visited.copy(), set(), 'found_goal', 'A* Path-Optimal', step_no)
                )
                return {'success': True, 'path': path, 'steps': step_no, 'algorithm': 'A* Path-Optimal'}

            for nxt in self.game.get_possible_moves(current):
                # Path-optimal: every move = 1 cost
                new_g = g + 1
                if nxt not in g_scores or new_g < g_scores[nxt]:
                    g_scores[nxt] = new_g
                    f_score = new_g + self.game.heuristic(nxt.position)
                    heapq.heappush(pq, (f_score, next(counter), new_g, nxt, path + [nxt]))

        self.visualization_steps.append(
            VisualizationStep(current, visited.copy(), set(), 'failed', 'A* Path-Optimal', step_no)
        )
        return {'success': False, 'path': [], 'steps': step_no, 'algorithm': 'A* Path-Optimal'}


    # ---------- Playback & UI updates ----------

    def _play(self):
        if not self.visualization_steps:
            return
        self.is_playing = True
        self.play_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self._tick()

    def _pause(self):
        self.is_playing = False
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)

    def _tick(self):
        if not self.is_playing:
            return
        if self.current_step < len(self.visualization_steps):
            self._step()
            self.play_speed = self.speed_var.get()
            self.root.after(self.play_speed, self._tick)
        else:
            self._pause()

    def _step(self):
        if self.current_step < len(self.visualization_steps):
            self.current_step += 1
            self._render_step()
            self._update_progress()

    def _reset(self):
        self._pause()
        self.current_step = 0
        self._draw_map()
        self._update_progress()
        self._update_map_info()

    def _render_step(self):
        if not self.visualization_steps or self.current_step == 0:
            return
        step = self.visualization_steps[self.current_step - 1]
        self.canvas.delete("highlight")

        for st in step.visited_states:
            self._highlight_cell(st.position.x, st.position.y, self.colors['visited'], "highlight")

        for st in step.frontier_states:
            self._highlight_cell(st.position.x, st.position.y, self.colors['frontier'], "highlight")

        self._highlight_cell(step.current_state.position.x, step.current_state.position.y,
                             self.colors['current'], "highlight")
        self._update_step_info(step)

        if hasattr(self, 'final_path') and self.current_step == len(self.visualization_steps):
            for state in self.final_path:
                self._highlight_cell(state.position.x, state.position.y, self.colors['path'], "highlight")


    def _update_step_info(self, step: VisualizationStep):
        self.step_info.delete(1.0, tk.END)
        info = (
            f"Algorithm: {step.algorithm}\n"
            f"Step: {step.step_number}\n"
            f"Status: {step.step_type}\n\n"
            f"Current Position: ({step.current_state.position.x}, {step.current_state.position.y})\n"
            f"Current Energy: {step.current_state.energy}\n"
            f"Foods Collected: {len(step.current_state.visited_foods)}\n\n"
            f"Total Visited: {len(step.visited_states)}\n"
            f"Frontier Size: {len(step.frontier_states)}\n"
        )
        self.step_info.insert(1.0, info)

    def _update_progress(self):
        total = len(self.visualization_steps)
        self.progress_bar.config(maximum=max(1, total))
        self.progress_var.set(self.current_step)
        self.progress_label.config(text=f"Step {self.current_step} of {total}")

    def _update_statistics(self, result, elapsed_ms: float):
        self.stats_info.delete(1.0, tk.END)
        path_len = len(result['path']) if result['success'] else 0
        final_energy = result['path'][-1].energy if result['success'] and result['path'] else 0

        txt = []
        txt.append("ALGORITHM RESULTS")
        txt.append("=" * 22)
        txt.append(f"Algorithm: {result['algorithm']}")
        txt.append(f"Map: {self.map_var.get()}")
        txt.append(f"Success: {'Yes' if result['success'] else 'No'}")
        txt.append(f"Collected Steps: {result['steps']}")
        txt.append(f"Elapsed: {elapsed_ms:.2f} ms")
        if result['success']:
            txt.append(f"Path Length: {path_len}")
            txt.append(f"Final Energy: {final_energy}")
        txt.append("")
        txt.append("=" * 22)
        txt.append("LEGEND:")
        txt.append("Red = Current Position")
        txt.append("Pink = Visited States")
        txt.append("Light Green = Frontier")
        txt.append("S = Start, T = Treasure")
        txt.append("F = Food, X = Obstacle")
        txt.append("~ = Swamp, ^ = Hills")

        self.stats_info.insert(1.0, "\n".join(txt))
    def draw_final_path(self, path):
        """Draw the final path from start to goal in magenta"""
        if not path:
            return
        for state in path:
            self.highlight_cell(state.position.x, state.position.y, self.colors['path'], "path")

    def _update_map_info(self):
        self.step_info.delete(1.0, tk.END)
        self.stats_info.delete(1.0, tk.END)
        init = self.game.get_initial_state()
        lines = [
            "MAP INFORMATION",
            "=" * 16,
            f"Size: {self.game.width} × {self.game.height}",
            f"Starting Energy: {init.energy}",
            f"Food Sources: {len(self.game.food_positions)}",
            f"Start: ({self.game.start_pos.x}, {self.game.start_pos.y})",
            f"Treasure: ({self.game.treasure_pos.x}, {self.game.treasure_pos.y})",
        ]
        self.stats_info.insert(1.0, "\n".join(lines))

    def _set_controls_busy(self, busy: bool):
        state = tk.DISABLED if busy else tk.NORMAL
        self.run_btn.config(state=tk.DISABLED if busy else tk.NORMAL)
        self.play_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED)
        self.step_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.DISABLED)

    def _set_controls_ready(self, has_result: bool):
        self.run_btn.config(state=tk.NORMAL)
        self.play_btn.config(state=tk.NORMAL if has_result else tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED)
        self.step_btn.config(state=tk.NORMAL if has_result else tk.DISABLED)
        self.reset_btn.config(state=tk.NORMAL)



def main():
    root = tk.Tk()
    app = TreasureHuntVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
