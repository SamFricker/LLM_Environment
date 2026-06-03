import ast
import json
import math
import random
import re
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import pygame

try:
    from chat import OPENROUTER_MODEL, client as LLM_CLIENT
    LLM_IMPORT_ERROR = ''
except Exception as exc:
    OPENROUTER_MODEL = 'unavailable'
    LLM_CLIENT = None
    LLM_IMPORT_ERROR = str(exc)


SCREEN_W, SCREEN_H = 1200, 720
FPS = 60

TILE_SIZE = 64
FOV = math.radians(70)
RENDER_RAYS = 180
OBS_RAYS = 48
MAX_DEPTH = 20.0

MOVE_SPEED = 0.125
TURN_SPEED = math.radians(15)

WORLD_W = 15
WORLD_H = 11
NUM_OBJECTS = 6
NUM_BARRIERS = 3
MAX_GENERATION_ATTEMPTS = 200

MAP_PANEL_W = 360
MAP_RADIUS = 8
PROMPT_RADIUS = 6
FONT_SIZE = 22
DEBUG_MINIMAP_SCALE = 10
AUTONOMY_ACTION_DELAY = 1
REPLAN_DELAY_FRAMES = 12
LLM_TIMEOUT_SECONDS = 20
LLM_MIN_BATCH_ACTIONS = 40
LLM_MAX_ACTIONS = 120
LOCAL_MACRO_MAX_ACTIONS = 240
LOCAL_SEARCH_MAX_ACTIONS = 80
LOOK_AROUND_ACTIONS = max(1, int(round((2 * math.pi) / TURN_SPEED)))

ALLOWED_ACTIONS = {'turn_left', 'turn_right', 'move_forward', 'move_back', 'interact', 'wait'}

# The model proposes moves, but the simulator and verifier get the final say.
LLM_SYSTEM_PROMPT = (
    "You control an embodied robot in a partially observed raycast grid world. "
    "Use only the observation provided: an explored occupancy map, a ray scan, inventory, "
    "and task progress. The map legend is ?: unknown, .: free, #: wall, @: you, B: bin/spawn, "
    "and object letters for collectible items. Heading 0 degrees points right/east; "
    "90 degrees points down/south. You may choose only primitive actions: turn_left, "
    "turn_right, move_forward, move_back, interact, wait. Interact picks up an object only "
    "when you are standing on it, and drops an item only when you are at the bin. "
    "Follow the user's verb strictly: go/find/move/navigate/stop means stop at the target without interacting; "
    "pick/grab/take/collect means interact with the target; bring/deliver/drop/bin/spawn means carry it to the bin. "
    "If the parsed intent lists excluded_objects, never pick up or deliver those objects. "
    "For explore-map tasks, keep exploring frontiers and stop when no reachable frontier remains. "
    "Color words refer to objects: lime/green=plate, blue=towel, orange=mug, pink=bowl, purple=spoon, yellow=lamp. "
    "Corner commands refer to the world/minimap frame, e.g. top right corner means the inner top-right floor cell. "
    "Return ONLY JSON in one of these schemas: "
    "{\"actions\":[\"move_forward\",\"move_forward\"],\"done\":false} "
    "or {\"actions\":[{\"action\":\"move_forward\",\"repeat\":12}],\"done\":false}. "
    f"Prefer {LLM_MIN_BATCH_ACTIONS} to {LLM_MAX_ACTIONS} primitive actions per response so the robot moves briskly. "
    "If a known path exists to the target or bin, return the whole route in one response. "
    "If more information is needed, search nearby frontiers efficiently; do not explore the whole map if a task target appears. "
    "Return fewer actions only when the robot is about to interact, very close to a wall, or genuinely uncertain. "
    "If the task is complete, return {\"actions\":[],\"done\":true}. "
    "Do not include markdown or explanation."
)


OBJECTS = {
    'm': ('mug', (255, 100, 0)),
    'p': ('plate', (23, 255, 31)),
    't': ('towel', (0, 140, 255)),
    'o': ('bowl', (255, 23, 232)),
    's': ('spoon', (120, 80, 200)),
    'l': ('lamp', (255, 210, 0)),
}

# The LLM and the map both use these short letters, so keep the aliases close by.
OBJECT_ALIASES = {
    'm': 'm',
    'mug': 'm',
    'orange': 'm',
    'p': 'p',
    'plate': 'p',
    'green': 'p',
    'lime': 'p',
    'limegreen': 'p',
    'lightgreen': 'p',
    't': 't',
    'towel': 't',
    'blue': 't',
    'cyan': 't',
    'azure': 't',
    'o': 'o',
    'bowl': 'o',
    'pink': 'o',
    'magenta': 'o',
    'fuchsia': 'o',
    's': 's',
    'spoon': 's',
    'purple': 's',
    'violet': 's',
    'l': 'l',
    'lamp': 'l',
    'yellow': 'l',
    'gold': 'l',
}


def normalize_angle(a: float) -> float:
    return a % (2 * math.pi)


def angle_delta(new_angle: float, old_angle: float) -> float:
    return (new_angle - old_angle + math.pi) % (2 * math.pi) - math.pi


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def apply_fog(color: Tuple[int, int, int], depth: float) -> Tuple[int, int, int]:
    fog = clamp(1.0 - (depth / MAX_DEPTH), 0.18, 1.0)
    return tuple(int(c * fog) for c in color)


def is_object_tile(ch: str) -> bool:
    return ch in OBJECTS


def odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def cell_of(value: float) -> int:
    return math.floor(value)


def reachable_from_spawn(grid: List[List[str]], spawn: Tuple[int, int]) -> Set[Tuple[int, int]]:
    h = len(grid)
    w = len(grid[0])
    sx, sy = spawn

    if grid[sy][sx] == '#':
        return set()

    q = deque([(sx, sy)])
    seen = {(sx, sy)}

    while q:
        x, y = q.popleft()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen:
                if grid[ny][nx] != '#':
                    seen.add((nx, ny))
                    q.append((nx, ny))

    return seen


def find_corner_candidates(grid: List[List[str]], reachable: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    h = len(grid)
    w = len(grid[0])
    spots = []

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if (x, y) not in reachable:
                continue
            if grid[y][x] != '.':
                continue

            n = grid[y - 1][x] == '#'
            s = grid[y + 1][x] == '#'
            e = grid[y][x + 1] == '#'
            wv = grid[y][x - 1] == '#'


            if (n and e) or (n and wv) or (s and e) or (s and wv):
                spots.append((x, y))

    return spots


def add_open_corner_barrier(grid: List[List[str]], x: int, y: int, orientation: str, horiz_len: int, vert_len: int):
    """Small L-shaped obstacle that stays open and does not attempt to seal areas."""
    h = len(grid)
    w = len(grid[0])

    def set_wall(px: int, py: int):
        if 0 < px < w - 1 and 0 < py < h - 1:
            grid[py][px] = '#'

    gap_index = 1

    if orientation == 'right_down':
        for i in range(horiz_len):
            if i != gap_index:
                set_wall(x + i, y)
        for j in range(vert_len):
            if j != gap_index:
                set_wall(x + horiz_len - 1, y + j)

    elif orientation == 'down_right':
        for j in range(vert_len):
            if j != gap_index:
                set_wall(x, y + j)
        for i in range(horiz_len):
            if i != gap_index:
                set_wall(x + i, y + vert_len - 1)

    elif orientation == 'left_down':
        for i in range(horiz_len):
            if i != gap_index:
                set_wall(x - i, y)
        for j in range(vert_len):
            if j != gap_index:
                set_wall(x - horiz_len + 1, y + j)

    elif orientation == 'down_left':
        for j in range(vert_len):
            if j != gap_index:
                set_wall(x, y + j)
        for i in range(horiz_len):
            if i != gap_index:
                set_wall(x - i, y + vert_len - 1)


def generate_simple_world(width: int, height: int, num_objects: int) -> List[str]:
    """Small room-like layout with reachable objects only."""
    width = odd(width)
    height = odd(height)

    # The room is simple on purpose; the challenge is the agent harness.
    for _attempt in range(MAX_GENERATION_ATTEMPTS):
        grid = [["."] * width for _ in range(height)]


        for x in range(width):
            grid[0][x] = '#'
            grid[height - 1][x] = '#'
        for y in range(height):
            grid[y][0] = '#'
            grid[y][width - 1] = '#'


        spawn = (1, 1)
        for yy in range(1, min(4, height - 1)):
            for xx in range(1, min(4, width - 1)):
                grid[yy][xx] = '.'

        # The barriers add corners for navigation without sealing off rooms.
        for _ in range(NUM_BARRIERS):
            x = random.randint(4, width - 5)
            y = random.randint(2, height - 4)
            horiz_len = random.randint(2, 4)
            vert_len = random.randint(2, 4)
            orientation = random.choice(['right_down', 'down_right', 'left_down', 'down_left'])
            add_open_corner_barrier(grid, x, y, orientation, horiz_len, vert_len)

        grid[spawn[1]][spawn[0]] = 'A'

        reachable = reachable_from_spawn(grid, spawn)
        candidates = find_corner_candidates(grid, reachable)
        candidates = [p for p in candidates if p != spawn]
        random.shuffle(candidates)

        letters = list(OBJECTS.keys())
        random.shuffle(letters)

        if len(candidates) < num_objects:
            fallback = []
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    if (x, y) in reachable and grid[y][x] == '.' and (x, y) != spawn:
                        n = grid[y - 1][x] == '#'
                        s = grid[y + 1][x] == '#'
                        e = grid[y][x + 1] == '#'
                        wv = grid[y][x - 1] == '#'
                        if n or s or e or wv:
                            fallback.append((x, y))
            random.shuffle(fallback)
            for p in fallback:
                if p not in candidates:
                    candidates.append(p)

        if len(candidates) < num_objects:
            continue

        n = min(num_objects, len(candidates), len(letters))
        for (x, y), ch in zip(candidates[:n], letters[:n]):
            grid[y][x] = ch


        reachable = reachable_from_spawn(grid, spawn)
        object_positions = [
            (x, y)
            for y in range(height)
            for x in range(width)
            if is_object_tile(grid[y][x])
        ]
        if all(pos in reachable for pos in object_positions):
            return ["".join(row) for row in grid]

    raise RuntimeError('Failed to generate a valid reachable map after many attempts.')


@dataclass
class Agent:
    x: float
    y: float
    angle: float
    carried_item: Optional[str] = None


class World:
    def __init__(self, map_rows: List[str]):
        if not map_rows:
            raise ValueError('Map is empty')

        width = len(map_rows[0])
        for row in map_rows:
            if len(row) != width:
                raise ValueError('All map rows must have the same length')

        self.map = [list(row) for row in map_rows]
        self.h = len(self.map)
        self.w = len(self.map[0])
        self.agent, self.spawn = self._find_agent_and_spawn()
        self.done = False
        self.delivered = 0
        self.total_items = sum(1 for row in self.map for ch in row if is_object_tile(ch))
        self.initial_items = {ch for row in self.map for ch in row if is_object_tile(ch)}
        self.delivered_items: Set[str] = set()
        self.message = 'Collect household items and return them to the bin at spawn.'

    def _find_agent_and_spawn(self) -> Tuple[Agent, Tuple[int, int]]:
        for y, row in enumerate(self.map):
            for x, ch in enumerate(row):
                if ch == 'A':
                    self.map[y][x] = '.'
                    return Agent(x + 0.5, y + 0.5, 0.0), (x, y)
        raise ValueError("No agent start 'A' found in map")

    def tile_at(self, mx: int, my: int) -> str:
        if mx < 0 or my < 0 or mx >= self.w or my >= self.h:
            return '#'
        return self.map[my][mx]

    def set_tile(self, mx: int, my: int, ch: str):
        if 0 <= mx < self.w and 0 <= my < self.h:
            self.map[my][mx] = ch

    def is_blocking(self, mx: int, my: int) -> bool:
        return self.tile_at(mx, my) == '#'

    def at_spawn(self) -> bool:
        return int(self.agent.x) == self.spawn[0] and int(self.agent.y) == self.spawn[1]

    def move(self, dx: float, dy: float):
        nx = self.agent.x + dx
        ny = self.agent.y + dy

        if not self.is_blocking(int(nx), int(self.agent.y)):
            self.agent.x = nx
        if not self.is_blocking(int(self.agent.x), int(ny)):
            self.agent.y = ny

    def step(self, action: str):
        if self.done:
            return

        action = action.lower().strip()

        if action == 'turn_left':
            self.agent.angle = normalize_angle(self.agent.angle - TURN_SPEED)
        elif action == 'turn_right':
            self.agent.angle = normalize_angle(self.agent.angle + TURN_SPEED)
        elif action == 'move_forward':
            self.move(math.cos(self.agent.angle) * MOVE_SPEED, math.sin(self.agent.angle) * MOVE_SPEED)
        elif action == 'move_back':
            self.move(-math.cos(self.agent.angle) * MOVE_SPEED, -math.sin(self.agent.angle) * MOVE_SPEED)
        elif action == 'interact':
            self.try_pickup_or_drop()
        elif action == 'wait':
            pass

        self.check_done()

    def try_pickup_or_drop(self):
        mx, my = int(self.agent.x), int(self.agent.y)
        tile = self.tile_at(mx, my)

        if self.agent.carried_item is not None:
            if self.at_spawn():
                dropped = self.agent.carried_item
                item_name = OBJECTS[dropped][0]
                self.delivered += 1
                self.delivered_items.add(dropped)
                self.message = f'Dropped {item_name} in the bin. ({self.delivered}/{self.total_items})'
                self.agent.carried_item = None
                self.check_done()
            else:
                self.message = 'Carry it back to the bin at spawn.'
            return

        if is_object_tile(tile):
            self.agent.carried_item = tile
            item_name = OBJECTS[tile][0]
            self.set_tile(mx, my, '.')
            self.message = f'Picked up {item_name}.'
        else:
            self.message = 'Nothing to pick up here.'

    def check_done(self):
        if self.delivered >= self.total_items and self.total_items > 0:
            self.done = True
            self.message = 'All items delivered. Success!'

    def get_local_view(self, radius: int = 2) -> List[List[str]]:
        ax, ay = int(self.agent.x), int(self.agent.y)
        view = []
        for dy in range(-radius, radius + 1):
            row = []
            for dx in range(-radius, radius + 1):
                row.append(self.tile_at(ax + dx, ay + dy))
            view.append(row)
        return view

    def get_observation(self) -> Dict:
        ax, ay = int(self.agent.x), int(self.agent.y)
        carried_name = None
        if self.agent.carried_item is not None:
            carried_name = OBJECTS[self.agent.carried_item][0]

        remaining = sum(1 for row in self.map for ch in row if is_object_tile(ch))

        return {
            'position': [round(self.agent.x, 2), round(self.agent.y, 2)],
            'grid_position': [ax, ay],
            'angle_rad': round(self.agent.angle, 3),
            'inventory': [carried_name] if carried_name else [],
            'local_view_5x5': self.get_local_view(radius=2),
            'goal': 'Collect household objects and deliver them to the bin at spawn.',
            'bin_location': list(self.spawn),
            'delivered': self.delivered,
            'delivered_items': [OBJECTS[ch][0] for ch in sorted(self.delivered_items)],
            'remaining': remaining + (1 if self.agent.carried_item is not None else 0),
            'message': self.message,
            'done': self.done,
        }


@dataclass
class Pose:
    x: float
    y: float
    angle: float


# Unknown cells are left out of the dict; it keeps the map cheap to grow.
class BeliefMap:
    """Sparse occupancy map in world coordinates."""

    def __init__(self):
        self.cells: Dict[Tuple[int, int], str] = {}
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0

    def _update_bounds(self, x: int, y: int):
        if not self.cells:
            self.min_x = self.max_x = x
            self.min_y = self.max_y = y
            return
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_y = max(self.max_y, y)

    def get_cell(self, x: int, y: int) -> str:
        return self.cells.get((x, y), '?')

    def set_cell(self, x: int, y: int, ch: str):
        existing = self.get_cell(x, y)
        if existing == ch:
            return

        if existing in OBJECTS and ch in {'.', '#'}:
            return
        if existing == '#' and ch == '.':
            return

        if ch in OBJECTS or ch in {'.', '#'}:
            self.cells[(x, y)] = ch
            self._update_bounds(x, y)

    def clear_object_at(self, x: int, y: int):
        if self.get_cell(x, y) in OBJECTS:
            self.cells[(x, y)] = '.'
            self._update_bounds(x, y)

    def mark_ray(self, pose: Pose, ray_angle: float, distance: float, hit: str):
        """Raycast footprint into the explored map."""
        free_len = max(0.0, distance - 0.12)
        steps = max(1, int(free_len / 0.05))
        for i in range(steps):
            d = i * 0.05
            x = pose.x + math.cos(ray_angle) * d
            y = pose.y + math.sin(ray_angle) * d
            self.set_cell(cell_of(x), cell_of(y), '.')

        hx = pose.x + math.cos(ray_angle) * distance
        hy = pose.y + math.sin(ray_angle) * distance
        cx, cy = cell_of(hx), cell_of(hy)
        if hit == '#':
            self.set_cell(cx, cy, '#')
        elif hit in OBJECTS:
            self.set_cell(cx, cy, hit)

    def update_from_scan(self, pose: Pose, scan: List[Dict]):
        for ray in scan:
            ang = pose.angle + ray['angle_offset']
            self.mark_ray(pose, ang, float(ray['distance']), ray['hit'])

    def window_text(self, center_x: int, center_y: int, radius: int) -> str:
        lines = []
        for y in range(center_y - radius, center_y + radius + 1):
            row = []
            for x in range(center_x - radius, center_x + radius + 1):
                row.append(self.get_cell(x, y))
            lines.append(''.join(row))
        return '\n'.join(lines)

    def count_known(self) -> Tuple[int, int, int, Dict[str, int]]:
        unknown = free = walls = 0
        objs: Dict[str, int] = {k: 0 for k in OBJECTS}
        if not self.cells:
            return unknown, free, walls, objs

        for y in range(self.min_y, self.max_y + 1):
            for x in range(self.min_x, self.max_x + 1):
                ch = self.get_cell(x, y)
                if ch == '?':
                    unknown += 1
                elif ch == '.':
                    free += 1
                elif ch == '#':
                    walls += 1
                elif ch in OBJECTS:
                    objs[ch] += 1
        return unknown, free, walls, objs


def cast_ray_truth(world: World, ray_angle: float) -> Tuple[float, str]:
    ox, oy = world.agent.x, world.agent.y
    start_cell = (int(ox), int(oy))
    sin_a = math.sin(ray_angle)
    cos_a = math.cos(ray_angle)

    depth = 0.0
    while depth < MAX_DEPTH:
        depth += 0.02
        x = ox + cos_a * depth
        y = oy + sin_a * depth
        cell = (int(x), int(y))
        t = world.tile_at(cell[0], cell[1])
        # Standing on an object should not make the camera blind.
        if t == '#' or (is_object_tile(t) and cell != start_cell):
            return depth, t
    return MAX_DEPTH, '?'


def draw_gradient_background(screen):
    for y in range(SCREEN_H // 2):
        t = y / (SCREEN_H // 2)
        c = (int(18 + 12 * t), int(18 + 12 * t), int(35 + 18 * t))
        pygame.draw.line(screen, c, (0, y), (SCREEN_W - MAP_PANEL_W, y))

    for y in range(SCREEN_H // 2, SCREEN_H):
        t = (y - SCREEN_H // 2) / (SCREEN_H // 2)
        c = (int(35 + 16 * t), int(28 + 10 * t), int(22 + 8 * t))
        pygame.draw.line(screen, c, (0, y), (SCREEN_W - MAP_PANEL_W, y))


def render_3d(screen, world: World):
    draw_gradient_background(screen)
    view_w = SCREEN_W - MAP_PANEL_W
    ray_angle = world.agent.angle - FOV / 2
    col_width = view_w / RENDER_RAYS

    for i in range(RENDER_RAYS):
        depth, tile = cast_ray_truth(world, ray_angle)
        corrected_depth = depth * math.cos(world.agent.angle - ray_angle)
        wall_h = int(min(SCREEN_H, (TILE_SIZE * 5) / max(corrected_depth, 0.0001)))
        y0 = SCREEN_H // 2 - wall_h // 2

        if tile == '#':
            base_color = (235, 235, 235)
        elif is_object_tile(tile):
            base_color = OBJECTS[tile][1]
        else:
            base_color = (200, 200, 200)

        shade_color = apply_fog(base_color, corrected_depth)
        x = int(i * col_width)
        w = max(1, int(math.ceil(col_width)))
        pygame.draw.rect(screen, shade_color, (x, y0, w, wall_h))
        edge = tuple(max(0, c - 40) for c in shade_color)
        pygame.draw.rect(screen, edge, (x, y0, 1, wall_h))

        ray_angle += FOV / RENDER_RAYS


def belief_tile_color(ch: str) -> Tuple[int, int, int]:
    if ch == '?':
        return (18, 22, 28)
    if ch == '.':
        return (58, 64, 72)
    if ch == '#':
        return (220, 220, 215)
    if ch in OBJECTS:
        return OBJECTS[ch][1]
    return (120, 120, 120)


def draw_legend_item(screen, font, x: int, y: int, color: Tuple[int, int, int], label: str):
    pygame.draw.rect(screen, color, (x, y + 2, 12, 12))
    pygame.draw.rect(screen, (120, 130, 145), (x, y + 2, 12, 12), 1)
    text = font.render(label, True, (230, 235, 240))
    screen.blit(text, (x + 18, y))


def render_explored_map(
    screen,
    belief_map: BeliefMap,
    belief_pose: Pose,
    goal_text: str,
    input_text: str,
    input_active: bool,
    last_status: str,
    show_truth_debug: bool,
    spawn: Tuple[int, int],
    planner_status: str = 'manual',
    queued_actions: int = 0,
    model_name: str = '',
):
    panel = pygame.Surface((MAP_PANEL_W, SCREEN_H), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 190))
    screen.blit(panel, (SCREEN_W - MAP_PANEL_W, 0))

    font = pygame.font.SysFont(None, 20)
    small = pygame.font.SysFont(None, 18)
    big = pygame.font.SysFont(None, 24)
    mono = pygame.font.SysFont('consolas', 17)

    cx, cy = cell_of(belief_pose.x), cell_of(belief_pose.y)
    start_x = SCREEN_W - MAP_PANEL_W + 10
    y = 12

    title = big.render('Explored map', True, (255, 255, 255))
    screen.blit(title, (start_x, y))
    y += 28

    cells_across = MAP_RADIUS * 2 + 1
    cell_size = min(18, (MAP_PANEL_W - 38) // cells_across)
    grid_size = cells_across * cell_size
    grid_x = start_x + (MAP_PANEL_W - 20 - grid_size) // 2
    grid_y = y

    visible_unknown = 0
    visible_free = 0
    visible_walls = 0
    visible_objects = 0

    for gy in range(cells_across):
        my = cy - MAP_RADIUS + gy
        for gx in range(cells_across):
            mx = cx - MAP_RADIUS + gx
            ch = belief_map.get_cell(mx, my)
            if ch == '?':
                visible_unknown += 1
            elif ch == '.':
                visible_free += 1
            elif ch == '#':
                visible_walls += 1
            elif ch in OBJECTS:
                visible_objects += 1

            rect = pygame.Rect(
                grid_x + gx * cell_size,
                grid_y + gy * cell_size,
                cell_size - 1,
                cell_size - 1,
            )
            pygame.draw.rect(screen, belief_tile_color(ch), rect)
            if ch in OBJECTS:
                label = mono.render(ch, True, (10, 10, 12))
                screen.blit(label, label.get_rect(center=rect.center))

            if (mx, my) == spawn:
                pygame.draw.rect(screen, (80, 160, 255), rect, 2)

    pygame.draw.rect(screen, (120, 130, 145), (grid_x - 1, grid_y - 1, grid_size + 1, grid_size + 1), 1)

    agent_px = grid_x + int((belief_pose.x - (cx - MAP_RADIUS)) * cell_size)
    agent_py = grid_y + int((belief_pose.y - (cy - MAP_RADIUS)) * cell_size)
    pygame.draw.circle(screen, (255, 75, 75), (agent_px, agent_py), max(4, cell_size // 3))
    facing_x = agent_px + int(math.cos(belief_pose.angle) * cell_size * 0.75)
    facing_y = agent_py + int(math.sin(belief_pose.angle) * cell_size * 0.75)
    pygame.draw.line(screen, (255, 230, 120), (agent_px, agent_py), (facing_x, facing_y), 2)

    y = grid_y + grid_size + 12
    known_total = len(belief_map.cells)
    unknown, free, walls, objs = belief_map.count_known()
    object_summary = ', '.join(f'{k}:{v}' for k, v in objs.items() if v) or 'none'

    info1 = font.render(f'belief pose: ({belief_pose.x:.2f}, {belief_pose.y:.2f})  {math.degrees(belief_pose.angle):.0f}deg', True, (255, 255, 255))
    info2 = font.render(f'known cells: {known_total}  bounds unknown: {unknown}', True, (230, 235, 240))
    info3 = font.render(f'free:{free} walls:{walls} objects:{object_summary}', True, (230, 235, 240))
    screen.blit(info1, (start_x, y)); y += 20
    screen.blit(info2, (start_x, y)); y += 20
    screen.blit(info3, (start_x, y)); y += 20

    y += 8
    status = font.render(last_status[:44], True, (255, 245, 170))
    screen.blit(status, (start_x, y)); y += 22
    goal = font.render(goal_text[:44], True, (210, 230, 255))
    screen.blit(goal, (start_x, y)); y += 22
    planner = font.render(f'{planner_status[:30]} | queue:{queued_actions}', True, (210, 230, 255))
    screen.blit(planner, (start_x, y)); y += 22
    if model_name:
        model = small.render(model_name[:44], True, (190, 200, 210))
        screen.blit(model, (start_x, y)); y += 18

    y += 8
    draw_legend_item(screen, small, start_x, y, belief_tile_color('?'), 'unknown')
    draw_legend_item(screen, small, start_x + 112, y, belief_tile_color('.'), 'free')
    draw_legend_item(screen, small, start_x + 202, y, belief_tile_color('#'), 'wall')
    y += 20
    draw_legend_item(screen, small, start_x, y, (255, 75, 75), 'belief pose')
    draw_legend_item(screen, small, start_x + 112, y, (80, 160, 255), 'bin/spawn')

    debug_text = 'on' if show_truth_debug else 'off'
    debug_line = small.render(f'truth minimap: {debug_text}', True, (200, 205, 215))
    screen.blit(debug_line, (start_x, y + 24))

    cmd_label = font.render('controls:', True, (255, 255, 255))
    screen.blit(cmd_label, (start_x, SCREEN_H - 84))
    controls = font.render('arrows move/turn | space interact | M debug', True, (255, 255, 255))
    screen.blit(controls, (start_x, SCREEN_H - 58))
    cmd_value = input_text if input_active else '(TAB for reset/stats)'
    cmd_text = font.render(f'command: {cmd_value}', True, (255, 255, 255))
    screen.blit(cmd_text, (start_x + 8, SCREEN_H - 36))


def draw_minimap(screen, world: World, belief_map: BeliefMap, belief_pose: Pose, x=10, y=10, scale=10):
    font = pygame.font.SysFont(None, 18)


    for my, row in enumerate(world.map):
        for mx, ch in enumerate(row):
            if ch == '#':
                c = (90, 90, 90)
            elif is_object_tile(ch):
                c = OBJECTS[ch][1]
            else:
                c = (30, 30, 30)
            pygame.draw.rect(screen, c, (x + mx * scale, y + my * scale, scale - 1, scale - 1))


    bx = x + world.spawn[0] * scale
    by = y + world.spawn[1] * scale
    pygame.draw.rect(screen, (60, 120, 220), (bx, by, scale - 1, scale - 1))

    ax = x + int(world.agent.x * scale)
    ay = y + int(world.agent.y * scale)
    pygame.draw.circle(screen, (220, 60, 60), (ax, ay), max(2, scale // 3))
    fx = ax + int(math.cos(world.agent.angle) * scale)
    fy = ay + int(math.sin(world.agent.angle) * scale)
    pygame.draw.line(screen, (255, 80, 80), (ax, ay), (fx, fy), 2)


    bpx = x + int(belief_pose.x * scale)
    bpy = y + int(belief_pose.y * scale)
    pygame.draw.circle(screen, (60, 220, 120), (bpx, bpy), max(2, scale // 4))
    bfx = bpx + int(math.cos(belief_pose.angle) * scale)
    bfy = bpy + int(math.sin(belief_pose.angle) * scale)
    pygame.draw.line(screen, (60, 220, 120), (bpx, bpy), (bfx, bfy), 2)

    carried = world.agent.carried_item
    label_text = 'carrying: none' if carried is None else f'carrying: {OBJECTS[carried][0]}'
    label = font.render(label_text, True, (255, 255, 255))
    screen.blit(label, (x, y + len(world.map) * scale + 4))


def update_belief_pose_from_input(belief_pose: Pose, action: str):
    if action == 'turn_left':
        belief_pose.angle = normalize_angle(belief_pose.angle - TURN_SPEED)
    elif action == 'turn_right':
        belief_pose.angle = normalize_angle(belief_pose.angle + TURN_SPEED)
    elif action == 'move_forward':
        belief_pose.x += math.cos(belief_pose.angle) * MOVE_SPEED
        belief_pose.y += math.sin(belief_pose.angle) * MOVE_SPEED
    elif action == 'move_back':
        belief_pose.x -= math.cos(belief_pose.angle) * MOVE_SPEED
        belief_pose.y -= math.sin(belief_pose.angle) * MOVE_SPEED


def update_belief_pose_from_odometry(belief_pose: Pose, before: Pose, world: World):
    belief_pose.x += world.agent.x - before.x
    belief_pose.y += world.agent.y - before.y
    belief_pose.angle = normalize_angle(belief_pose.angle + angle_delta(world.agent.angle, before.angle))


def get_ray_scan(world: World) -> List[Dict]:
    scan = []
    for i in range(OBS_RAYS):
        t = 0 if OBS_RAYS == 1 else i / (OBS_RAYS - 1)
        offset = -FOV / 2 + t * FOV
        depth, hit = cast_ray_truth(world, world.agent.angle + offset)
        scan.append({
            'ray': i,
            'angle_offset': offset,
            'angle_deg': round(math.degrees(offset), 1),
            'distance': round(depth, 3),
            'hit': hit,
        })
    return scan


def update_belief_map_from_scan(world: World, belief_map: BeliefMap, belief_pose: Pose) -> List[Dict]:

    # Truth rays are projected into the robot's current belief frame.
    scan = get_ray_scan(world)
    belief_map.update_from_scan(belief_pose, scan)
    return scan


def move_action_from_keys(keys) -> Optional[str]:
    if keys[pygame.K_LEFT]:
        return 'turn_left'
    if keys[pygame.K_RIGHT]:
        return 'turn_right'
    if keys[pygame.K_UP]:
        return 'move_forward'
    if keys[pygame.K_DOWN]:
        return 'move_back'
    return None


def prompt_map_text(belief_map: BeliefMap, belief_pose: Pose, spawn: Tuple[int, int], radius: int = PROMPT_RADIUS) -> str:
    cx, cy = cell_of(belief_pose.x), cell_of(belief_pose.y)
    ax, ay = cx, cy
    lines = []
    for y in range(cy - radius, cy + radius + 1):
        row = []
        for x in range(cx - radius, cx + radius + 1):
            if (x, y) == (ax, ay):
                row.append('@')
            elif (x, y) == spawn:
                row.append('B')
            else:
                row.append(belief_map.get_cell(x, y))
        lines.append(''.join(row))
    return '\n'.join(lines)


def compact_scan(scan: List[Dict]) -> List[Dict]:
    return [
        {
            'angle_deg': ray['angle_deg'],
            'distance': round(float(ray['distance']), 2),
            'hit': ray['hit'],
        }
        for ray in scan
    ]


def object_catalog() -> Dict[str, str]:
    return {letter: name for letter, (name, _color) in OBJECTS.items()}


def build_llm_prompt(goal: str, world: World, belief_map: BeliefMap, belief_pose: Pose, scan: List[Dict]) -> str:
    carried = None
    if world.agent.carried_item is not None:
        carried = OBJECTS[world.agent.carried_item][0]

    # Keep this observation boring and explicit; it makes model outputs easier to audit.
    unknown, free, walls, objs = belief_map.count_known()
    excluded = excluded_objects_from_goal(goal)
    targets = target_objects_from_goal(goal)
    target = target_object_from_goal(goal)
    corner_target = corner_target_from_goal(goal, world)
    observation = {
        'user_goal': goal,
        'parsed_intent': {
            'target_object': OBJECTS[target][0] if target is not None else None,
            'target_objects': [OBJECTS[ch][0] for ch in targets],
            'corner_target': list(corner_target[:2]) if corner_target is not None else None,
            'excluded_objects': [OBJECTS[ch][0] for ch in sorted(excluded)],
            'navigate_only': wants_navigation_only(goal),
            'pickup_target': wants_pickup(goal),
            'deliver_to_bin': wants_bin_or_delivery(goal),
            'collect_multiple': wants_collection_goal(goal) or len(targets) > 1,
            'explore_map': wants_exploration_goal(goal),
            'spin_once': wants_spin_goal(goal),
        },
        'action_space': sorted(ALLOWED_ACTIONS),
        'robot': {
            'belief_position': [round(belief_pose.x, 2), round(belief_pose.y, 2)],
            'belief_cell': [cell_of(belief_pose.x), cell_of(belief_pose.y)],
            'heading_degrees': round(math.degrees(belief_pose.angle), 1),
            'inventory': [carried] if carried else [],
            'bin_cell': list(world.spawn),
        },
        'task_progress': {
            'delivered': world.delivered,
            'delivered_items': [OBJECTS[ch][0] for ch in sorted(world.delivered_items)],
            'total_items': world.total_items,
            'remaining_visible_or_uncollected': sum(1 for row in world.map for ch in row if is_object_tile(ch))
            + (1 if world.agent.carried_item is not None else 0),
            'last_environment_message': world.message,
            'world_done': world.done,
        },
        'symbols': object_catalog(),
        'map_stats': {
            'known_cells': len(belief_map.cells),
            'free_cells': free,
            'wall_cells': walls,
            'unknown_cells_inside_known_bounds': unknown,
            'objects_in_belief_map': {k: v for k, v in objs.items() if v},
        },
        'explored_map_centered_on_robot': prompt_map_text(belief_map, belief_pose, world.spawn),
        'ray_scan_left_to_right': compact_scan(scan),
    }

    return (
        "Choose the next batched action sequence for the robot.\n"
        "Prefer progress toward the user's goal. If the target is not visible in the explored map, "
        "explore unknown space safely using the scan and map. Avoid repeatedly pushing into walls. "
        f"When there is known free space ahead, batch repeated movement and return {LLM_MIN_BATCH_ACTIONS} "
        f"to {LLM_MAX_ACTIONS} primitive actions before asking for another observation. "
        "The simulator will still validate collisions after every primitive action.\n\n"
        f"Observation JSON:\n{json.dumps(observation, indent=2)}"
    )


def parse_jsonish(text: str) -> Optional[Any]:
    stripped = text.strip()
    if stripped.startswith('```'):
        stripped = stripped.strip('`').strip()
        if stripped.lower().startswith('json'):
            stripped = stripped[4:].strip()

    candidates = [stripped]
    first = stripped.find('{')
    last = stripped.rfind('}')
    if first != -1 and last != -1 and last > first:
        candidates.append(stripped[first:last + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            return ast.literal_eval(candidate)
        except Exception:
            pass
    return None


def append_action(clean: List[str], action: Any, repeat: int = 1):
    if not isinstance(action, str):
        return
    normalized = action.strip().lower()
    if normalized not in ALLOWED_ACTIONS:
        return

    repeat = int(clamp(repeat, 1, LLM_MAX_ACTIONS))
    for _ in range(repeat):
        if len(clean) >= LLM_MAX_ACTIONS:
            return
        clean.append(normalized)


def sanitize_actions(actions: List[Any]) -> List[str]:
    clean = []
    for item in actions:
        if isinstance(item, str):
            append_action(clean, item)
        elif isinstance(item, dict):
            repeat = item.get('repeat', 1)
            try:
                repeat = int(repeat)
            except Exception:
                repeat = 1
            append_action(clean, item.get('action'), repeat)
        if len(clean) >= LLM_MAX_ACTIONS:
            break
    return clean


def parse_llm_response(text: str) -> Tuple[List[str], bool]:
    payload = parse_jsonish(text)
    if payload is None:
        return [], False

    if isinstance(payload, list):
        return sanitize_actions(payload), False

    if not isinstance(payload, dict):
        return [], False

    done = bool(payload.get('done', False))
    if isinstance(payload.get('actions'), list):
        return sanitize_actions(payload['actions']), done

    if isinstance(payload.get('action'), str):
        repeat = payload.get('repeat', 1)
        try:
            repeat = int(repeat)
        except Exception:
            repeat = 1
        repeat = int(clamp(repeat, 1, LLM_MAX_ACTIONS))
        return sanitize_actions([payload['action']] * repeat), done

    return [], done


def goal_words(goal: str) -> Set[str]:
    return set(re.findall(r'[a-z0-9]+', goal.lower()))


# A small parser catches the phrases the local planner can handle deterministically.
def object_mentions_in_text(text: str) -> List[str]:
    mentions: List[Tuple[int, str]] = []
    for match in re.finditer(r'[a-z0-9]+', text.lower()):
        letter = OBJECT_ALIASES.get(match.group(0))
        if letter is not None:
            mentions.append((match.start(), letter))

    ordered: List[str] = []
    seen: Set[str] = set()
    for _pos, letter in sorted(mentions):
        if letter not in seen:
            ordered.append(letter)
            seen.add(letter)
    return ordered


def object_aliases_in_text(text: str) -> Set[str]:
    return set(object_mentions_in_text(text))


def excluded_objects_from_goal(goal: str) -> Set[str]:
    text = goal.lower()
    excluded: Set[str] = set()
    markers = [
        'apart from',
        'except for',
        'except',
        'excluding',
        'other than',
        'but not',
        'leave',
    ]

    for marker in markers:
        idx = text.find(marker)
        if idx == -1:
            continue
        excluded.update(object_aliases_in_text(text[idx + len(marker):]))

    return excluded


def target_object_from_goal(goal: str) -> Optional[str]:
    targets = target_objects_from_goal(goal)
    return targets[0] if targets else None


def target_objects_from_goal(goal: str) -> List[str]:
    excluded = excluded_objects_from_goal(goal)
    return [letter for letter in object_mentions_in_text(goal) if letter not in excluded]


def has_ordered_targets(goal: str) -> bool:
    words = goal_words(goal)
    return bool(words & {'first', 'then', 'next', 'before', 'after'})


def wants_bin_or_delivery(goal: str) -> bool:
    words = goal_words(goal)
    destination_words = {'bin', 'spawn', 'home', 'back'}
    direct_words = {'return', 'deliver', 'drop'}
    return bool(words & direct_words) or bool(words & destination_words) or ('bring' in words and bool(words & destination_words))


def wants_pickup(goal: str) -> bool:
    words = goal_words(goal)
    return bool(words & {'collect', 'pickup', 'pick', 'grab', 'take', 'get'})


def wants_navigation_only(goal: str) -> bool:
    words = goal_words(goal)
    navigation_words = {'go', 'goto', 'move', 'navigate', 'walk', 'find', 'approach', 'visit'}
    stop_words = {'stop', 'stay', 'wait', 'only'}
    return bool(words & navigation_words) and (bool(words & stop_words) or not wants_pickup(goal)) and not wants_bin_or_delivery(goal)


def corner_target_from_goal(goal: str, world: World) -> Optional[Tuple[int, int, str]]:
    words = goal_words(goal)
    has_corner_intent = 'corner' in words or bool(words & {'topright', 'topleft', 'bottomright', 'bottomleft'})
    if not has_corner_intent:
        return None

    top = bool(words & {'top', 'north', 'upper', 'up'}) or 'topright' in words or 'topleft' in words
    bottom = bool(words & {'bottom', 'south', 'lower', 'down'}) or 'bottomright' in words or 'bottomleft' in words
    left = bool(words & {'left', 'west'}) or 'topleft' in words or 'bottomleft' in words
    right = bool(words & {'right', 'east'}) or 'topright' in words or 'bottomright' in words

    if top == bottom or left == right:
        return None

    x = world.w - 2 if right else 1
    y = world.h - 2 if bottom else 1
    vertical = 'bottom' if bottom else 'top'
    horizontal = 'right' if right else 'left'
    return x, y, f'{vertical} {horizontal} corner'


def wants_collection_goal(goal: str) -> bool:
    words = goal_words(goal)
    if target_object_from_goal(goal) is not None:
        return False
    return bool(words & {'collect', 'objects', 'items', 'household', 'all', 'them', 'everything', 'clean', 'tidy'})


def wants_exploration_goal(goal: str) -> bool:
    words = goal_words(goal)
    return bool(words & {'explore', 'map', 'scout', 'survey'}) and not wants_collection_goal(goal)


def wants_spin_goal(goal: str) -> bool:
    text = goal.lower()
    words = goal_words(goal)
    return (
        bool(words & {'spin', 'rotate'})
        or 'turn around' in text
        or 'circle once' in text
        or '360' in words
        or 'look around' in text
    ) and not wants_search_goal(goal)


def spin_actions_from_goal(goal: str) -> List[str]:
    words = goal_words(goal)
    action = 'turn_left' if bool(words & {'left', 'counterclockwise', 'anticlockwise'}) else 'turn_right'
    repeats = 2 if bool(words & {'twice', 'two', '2'}) else 1
    return [action] * (LOOK_AROUND_ACTIONS * repeats)


def wants_search_goal(goal: str) -> bool:
    return target_object_from_goal(goal) is not None or wants_collection_goal(goal) or wants_exploration_goal(goal)


def goal_satisfied_by_state(goal: str, world: World, belief_map: Optional[BeliefMap] = None) -> bool:
    # Completion is checked from state, not from whether the model sounds confident.
    if world.done:
        return True

    if wants_exploration_goal(goal):
        return belief_map is not None and not frontier_cells(belief_map)

    excluded = excluded_objects_from_goal(goal)
    corner_target = corner_target_from_goal(goal, world)
    if corner_target is not None:
        return (int(world.agent.x), int(world.agent.y)) == (corner_target[0], corner_target[1])

    if wants_collection_goal(goal):
        required = {ch for ch in world.initial_items if ch not in excluded}
        carrying_required = world.agent.carried_item is not None and world.agent.carried_item in required
        carrying_excluded = world.agent.carried_item is not None and world.agent.carried_item in excluded
        return required.issubset(world.delivered_items) and not carrying_required and not carrying_excluded

    targets = target_objects_from_goal(goal)
    if len(targets) > 1:
        requested = set(targets)
        at_requested_target = world.tile_at(int(world.agent.x), int(world.agent.y)) in requested

        if wants_navigation_only(goal) and at_requested_target:
            return True

        if wants_bin_or_delivery(goal):
            if any(ch not in world.initial_items for ch in requested):
                return False
            carrying_requested = world.agent.carried_item is not None and world.agent.carried_item in requested
            return requested.issubset(world.delivered_items) and not carrying_requested

        return world.agent.carried_item in requested

    target = targets[0] if targets else None
    if target is None:
        return False

    if target not in world.initial_items and target not in world.delivered_items:
        return False

    target_remaining = any(ch == target for row in world.map for ch in row)
    at_target = world.tile_at(int(world.agent.x), int(world.agent.y)) == target

    if wants_navigation_only(goal) and at_target:
        return True

    if wants_bin_or_delivery(goal):
        return target in world.delivered_items

    if world.agent.carried_item == target and not wants_bin_or_delivery(goal):
        return True

    if world.agent.carried_item != target and not target_remaining:
        return target in world.delivered_items

    return False


def should_deliver_after_pickup(goal: str) -> bool:
    return wants_bin_or_delivery(goal) or wants_collection_goal(goal)


def should_pickup_target(goal: str) -> bool:
    return wants_pickup(goal) or should_deliver_after_pickup(goal)


def is_belief_walkable(belief_map: BeliefMap, cell: Tuple[int, int], extra_walkable: Set[Tuple[int, int]]) -> bool:
    if cell in extra_walkable:
        return True
    ch = belief_map.get_cell(cell[0], cell[1])
    return ch == '.' or ch in OBJECTS


def belief_neighbors(belief_map: BeliefMap, cell: Tuple[int, int], extra_walkable: Set[Tuple[int, int]]):
    x, y = cell
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nxt = (x + dx, y + dy)
        if is_belief_walkable(belief_map, nxt, extra_walkable):
            yield nxt


def bfs_path(
    belief_map: BeliefMap,
    start: Tuple[int, int],
    goals: Set[Tuple[int, int]],
    extra_walkable: Optional[Set[Tuple[int, int]]] = None,
) -> Optional[List[Tuple[int, int]]]:
    if not goals:
        return None

    extra = extra_walkable or set()
    q = deque([start])
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

    while q:
        current = q.popleft()
        if current in goals:
            path = [current]
            while came_from[current] is not None:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for nxt in belief_neighbors(belief_map, current, extra):
            if nxt in came_from:
                continue
            came_from[nxt] = current
            q.append(nxt)

    return None


def truth_bfs_path(world: World, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    # Corner commands refer to the minimap frame, so they use the known world grid.
    q = deque([start])
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

    while q:
        current = q.popleft()
        if current == goal:
            path = [current]
            while came_from[current] is not None:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        x, y = current
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nxt = (x + dx, y + dy)
            if nxt in came_from or world.is_blocking(nxt[0], nxt[1]):
                continue
            came_from[nxt] = current
            q.append(nxt)

    return None


def find_belief_object(belief_map: BeliefMap, target: str) -> Optional[Tuple[int, int]]:
    for cell, ch in belief_map.cells.items():
        if ch == target:
            return cell
    return None


def known_object_cells(belief_map: BeliefMap, excluded: Optional[Set[str]] = None) -> Dict[str, Tuple[int, int]]:
    excluded = excluded or set()
    objects = {}
    for cell, ch in belief_map.cells.items():
        if ch in OBJECTS and ch not in excluded:
            objects[ch] = cell
    return objects


def frontier_cells(belief_map: BeliefMap) -> List[Tuple[int, int]]:
    frontiers = []
    for (x, y), ch in belief_map.cells.items():
        if ch != '.' and ch not in OBJECTS:
            continue
        if any(belief_map.get_cell(x + dx, y + dy) == '?' for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]):
            frontiers.append((x, y))
    return frontiers


def is_frontier_cell(belief_map: BeliefMap, cell: Tuple[int, int]) -> bool:
    x, y = cell
    ch = belief_map.get_cell(x, y)
    if ch != '.' and ch not in OBJECTS:
        return False
    return any(belief_map.get_cell(x + dx, y + dy) == '?' for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)])


def choose_exploration_goal(belief_map: BeliefMap, start: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    candidates = frontier_cells(belief_map)
    if not candidates:
        return None


    best_path = None
    best_score = -10**9
    for goal in candidates:
        path = bfs_path(belief_map, start, {goal})
        if path is None:
            continue
        unknown_edges = 0
        x, y = goal
        unknown_edges += sum(
            1
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
            if belief_map.get_cell(x + dx, y + dy) == '?'
        )
        score = unknown_edges * 6 - len(path)
        if score > best_score:
            best_score = score
            best_path = path

    return best_path[-1] if best_path else None


def turn_actions_to_heading(current_angle: float, desired_angle: float) -> Tuple[List[str], float]:
    diff = (desired_angle - current_angle + math.pi) % (2 * math.pi) - math.pi
    steps = int(round(diff / TURN_SPEED))
    actions = []
    if steps > 0:
        actions.extend(['turn_right'] * steps)
    elif steps < 0:
        actions.extend(['turn_left'] * abs(steps))
    return actions, normalize_angle(current_angle + steps * TURN_SPEED)


def path_to_actions(path: List[Tuple[int, int]], start_angle: float, max_actions: int = LOCAL_MACRO_MAX_ACTIONS) -> List[str]:
    if not path or len(path) < 2:
        return []

    # Convert a cell path into the same small actions the LLM is allowed to use.
    actions: List[str] = []
    angle = start_angle
    moves_per_cell = max(1, int(round(1.0 / MOVE_SPEED)))
    heading_for_delta = {
        (1, 0): 0.0,
        (0, 1): math.pi / 2,
        (-1, 0): math.pi,
        (0, -1): 3 * math.pi / 2,
    }

    for current, nxt in zip(path, path[1:]):
        dx = nxt[0] - current[0]
        dy = nxt[1] - current[1]
        desired = heading_for_delta.get((dx, dy))
        if desired is None:
            break

        turns, angle = turn_actions_to_heading(angle, desired)
        for action in turns + ['move_forward'] * moves_per_cell:
            if len(actions) >= max_actions:
                return actions
            actions.append(action)

    return actions


def nearest_known_object_path(
    belief_map: BeliefMap,
    start: Tuple[int, int],
    extra_walkable: Set[Tuple[int, int]],
    excluded: Optional[Set[str]] = None,
) -> Tuple[Optional[str], Optional[List[Tuple[int, int]]]]:
    objects = known_object_cells(belief_map, excluded)
    if not objects:
        return None, None

    path = bfs_path(belief_map, start, set(objects.values()), extra_walkable | set(objects.values()))
    if path is None:
        return None, None

    target_cell = path[-1]
    for letter, cell in objects.items():
        if cell == target_cell:
            return letter, path

    return None, None


def nearest_known_target_path(
    belief_map: BeliefMap,
    start: Tuple[int, int],
    extra_walkable: Set[Tuple[int, int]],
    targets: List[str],
) -> Tuple[Optional[str], Optional[List[Tuple[int, int]]]]:
    cells: Dict[str, Tuple[int, int]] = {}
    for target in targets:
        cell = find_belief_object(belief_map, target)
        if cell is not None:
            cells[target] = cell

    if not cells:
        return None, None

    path = bfs_path(belief_map, start, set(cells.values()), extra_walkable | set(cells.values()))
    if path is None:
        return None, None

    target_cell = path[-1]
    for letter, cell in cells.items():
        if cell == target_cell:
            return letter, path

    return None, None


def append_return_to_bin(
    actions: List[str],
    belief_map: BeliefMap,
    from_cell: Tuple[int, int],
    start_angle: float,
    spawn: Tuple[int, int],
    extra_walkable: Set[Tuple[int, int]],
) -> List[str]:
    if len(actions) >= LOCAL_MACRO_MAX_ACTIONS:
        return actions

    back_path = bfs_path(belief_map, from_cell, {spawn}, extra_walkable | {from_cell, spawn})
    if not back_path:
        return actions

    return_angle = simulate_action_heading(start_angle, actions)
    remaining_budget = LOCAL_MACRO_MAX_ACTIONS - len(actions)
    actions.extend(path_to_actions(back_path, return_angle, remaining_budget))
    if len(actions) < LOCAL_MACRO_MAX_ACTIONS:
        actions.append('interact')
    return actions


def local_macro_plan(goal: str, world: World, belief_map: BeliefMap, belief_pose: Pose) -> Tuple[List[str], str]:
    start = (cell_of(belief_pose.x), cell_of(belief_pose.y))
    extra_walkable = {start, world.spawn}
    deliver_after_pickup = should_deliver_after_pickup(goal)
    excluded = excluded_objects_from_goal(goal)
    targets = target_objects_from_goal(goal)

    # Common tasks are handled here so the API is not asked to micromanage movement.
    if wants_spin_goal(goal):
        return spin_actions_from_goal(goal), 'one-shot spin'

    corner_target = corner_target_from_goal(goal, world)
    if corner_target is not None:
        target_cell = (corner_target[0], corner_target[1])
        path = truth_bfs_path(world, start, target_cell)
        if path:
            actions = path_to_actions(path, belief_pose.angle)
            if actions:
                return actions, f'local path to {corner_target[2]}'
            return [], 'already at corner'

    if world.agent.carried_item is not None and deliver_after_pickup:
        if world.agent.carried_item in excluded:
            return [], 'holding excluded item'
        if targets and world.agent.carried_item not in targets:
            return [], 'holding unrelated item'
        path = bfs_path(belief_map, start, {world.spawn}, extra_walkable)
        if path:
            actions = path_to_actions(path, belief_pose.angle)
            if len(actions) < LOCAL_MACRO_MAX_ACTIONS:
                actions.append('interact')
            return actions, 'local path to bin'

    if len(targets) > 1:
        pending_targets = [target for target in targets if target not in world.delivered_items]
        if has_ordered_targets(goal) and pending_targets:
            pending_targets = pending_targets[:1]

        object_letter, object_path = nearest_known_target_path(belief_map, start, extra_walkable, pending_targets)
        if object_letter is not None and object_path:
            object_cell = object_path[-1]
            actions = path_to_actions(object_path, belief_pose.angle)
            pickup_target = should_pickup_target(goal)
            if pickup_target and len(actions) < LOCAL_MACRO_MAX_ACTIONS:
                actions.append('interact')
            if deliver_after_pickup:
                append_return_to_bin(actions, belief_map, object_cell, belief_pose.angle, world.spawn, extra_walkable)
            reason = f'local path to {OBJECTS[object_letter][0]}'
            if pickup_target and deliver_after_pickup:
                reason = f'local deliver {OBJECTS[object_letter][0]}'
            elif pickup_target:
                reason = f'local pick up {OBJECTS[object_letter][0]}'
            return actions, reason

    target = targets[0] if targets else None
    if target is not None:
        target_cell = find_belief_object(belief_map, target)
        if target_cell is not None:
            path = bfs_path(belief_map, start, {target_cell}, extra_walkable | {target_cell})
            if path:
                actions = path_to_actions(path, belief_pose.angle)
                pickup_target = should_pickup_target(goal)
                # "Go to" and "pick up" are intentionally different commands.
                if pickup_target and len(actions) < LOCAL_MACRO_MAX_ACTIONS:
                    actions.append('interact')
                if deliver_after_pickup:
                    append_return_to_bin(actions, belief_map, target_cell, belief_pose.angle, world.spawn, extra_walkable)
                reason = f'local path to {OBJECTS[target][0]}'
                if pickup_target and deliver_after_pickup:
                    reason = f'local deliver {OBJECTS[target][0]}'
                elif pickup_target:
                    reason = f'local pick up {OBJECTS[target][0]}'
                return actions, reason

    if target is None and wants_collection_goal(goal):
        object_letter, object_path = nearest_known_object_path(belief_map, start, extra_walkable, excluded)
        if object_letter is not None and object_path:
            object_cell = object_path[-1]
            actions = path_to_actions(object_path, belief_pose.angle)
            if len(actions) < LOCAL_MACRO_MAX_ACTIONS:
                actions.append('interact')
            append_return_to_bin(actions, belief_map, object_cell, belief_pose.angle, world.spawn, extra_walkable)
            return actions, f'local collect {OBJECTS[object_letter][0]}'

    frontier = choose_exploration_goal(belief_map, start)
    if frontier is not None:
        path = bfs_path(belief_map, start, {frontier}, extra_walkable)
        if path:
            actions = path_to_actions(path, belief_pose.angle, LOCAL_SEARCH_MAX_ACTIONS)
            if actions:
                return actions, 'targeted search frontier'
            if path[-1] == start and is_frontier_cell(belief_map, start):
                # If we are already on the edge of the known map, rotate and gather rays.
                return ['turn_right'] * LOOK_AROUND_ACTIONS, 'look around frontier'

    if wants_search_goal(goal):
        if is_frontier_cell(belief_map, start):
            return ['turn_right'] * LOOK_AROUND_ACTIONS, 'look around frontier'
        return [], 'search exhausted'

    return [], 'no local macro available'


def should_interrupt_for_better_task_plan(goal: str, planner_status: str, world: World, belief_map: BeliefMap) -> bool:
    status = planner_status.lower()
    interruptible = 'search' in status or 'frontier' in status or 'api' in status
    if not interruptible:
        return False

    excluded = excluded_objects_from_goal(goal)

    if world.agent.carried_item is not None and world.agent.carried_item not in excluded and should_deliver_after_pickup(goal):
        return True

    targets = target_objects_from_goal(goal)
    if len(targets) > 1:
        pending_targets = [target for target in targets if target not in world.delivered_items]
        if has_ordered_targets(goal) and pending_targets:
            pending_targets = pending_targets[:1]
        return any(find_belief_object(belief_map, target) is not None for target in pending_targets)

    target = targets[0] if targets else None
    if target is not None:
        return find_belief_object(belief_map, target) is not None

    if wants_collection_goal(goal):
        return bool(known_object_cells(belief_map, excluded))

    return False


def simulate_action_heading(start_angle: float, actions: List[str]) -> float:
    angle = start_angle
    for action in actions:
        if action == 'turn_left':
            angle = normalize_angle(angle - TURN_SPEED)
        elif action == 'turn_right':
            angle = normalize_angle(angle + TURN_SPEED)
    return angle


def call_llm_planner(prompt: str) -> Dict[str, Any]:
    if LLM_CLIENT is None:
        return {
            'actions': [],
            'done': False,
            'raw': '',
            'error': f'LLM unavailable: {LLM_IMPORT_ERROR or "client not configured"}',
        }

    try:
        response = LLM_CLIENT.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {'role': 'system', 'content': LLM_SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.1,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        raw = response.choices[0].message.content or ''
        actions, done = parse_llm_response(raw)
        return {'actions': actions, 'done': done, 'raw': raw, 'error': ''}
    except Exception as exc:
        return {'actions': [], 'done': False, 'raw': '', 'error': str(exc)}


def main():
    pygame.init()
    pygame.display.set_caption('LLM Environment')
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, FONT_SIZE)
    small = pygame.font.SysFont(None, 18)

    map_rows = generate_simple_world(WORLD_W, WORLD_H, NUM_OBJECTS)
    world = World(map_rows)


    belief_pose = Pose(world.spawn[0] + 0.5, world.spawn[1] + 0.5, 0.0)
    belief_map = BeliefMap()


    last_scan = update_belief_map_from_scan(world, belief_map, belief_pose)

    input_active = True
    input_text = ''
    last_status = 'Type a task like: find the mug and bring it to the bin'
    action_repeat_delay = 0
    autonomy_action_delay = 0
    replan_delay = 0
    show_truth_debug = False
    space_was_down = False
    active_goal = ''
    action_queue: List[str] = []
    planner_status = 'waiting for task'
    agent_paused = False
    last_llm_reply = ''
    planner_generation = 0
    planning_generation = 0
    planning_future = None
    planner_executor = ThreadPoolExecutor(max_workers=2)

    # Planning runs off-thread so the raycast view keeps updating.
    def shutdown_and_exit():
        planner_executor.shutdown(wait=False, cancel_futures=True)
        pygame.quit()
        sys.exit()

    def complete_active_goal(reason: str) -> bool:
        nonlocal active_goal, action_queue, agent_paused, planner_generation, planning_future, planner_status, last_status, replan_delay
        if not active_goal.strip():
            return False
        one_shot_done = planner_status.startswith('one-shot') and not action_queue
        if not one_shot_done and not goal_satisfied_by_state(active_goal, world, belief_map):
            return False

        # This is the guard that stops old queued moves after a task has succeeded.
        action_queue.clear()
        agent_paused = False
        planner_generation += 1
        if planning_future is not None and not planning_future.done():
            planning_future.cancel()
        planning_future = None
        replan_delay = 0
        planner_status = 'success'
        last_status = 'SUCCESS: one-shot action complete' if one_shot_done else f'SUCCESS: {reason}'
        active_goal = ''
        return True

    def apply_action(action: str):
        nonlocal last_scan, last_status
        before = Pose(world.agent.x, world.agent.y, world.agent.angle)
        before_carried = world.agent.carried_item

        world.step(action)
        update_belief_pose_from_odometry(belief_pose, before, world)

        if action == 'interact' and before_carried is None and world.agent.carried_item is not None:
            belief_map.clear_object_at(cell_of(belief_pose.x), cell_of(belief_pose.y))

        last_scan = update_belief_map_from_scan(world, belief_map, belief_pose)
        last_status = world.message
        complete_active_goal(world.message)

    def clear_agent(status: str):
        nonlocal active_goal, action_queue, agent_paused, planner_generation, planning_future, planner_status, last_status
        active_goal = ''
        action_queue.clear()
        agent_paused = False
        planner_generation += 1
        if planning_future is not None and not planning_future.done():
            planning_future.cancel()
        planning_future = None
        planner_status = 'stopped'
        last_status = status

    def kick_off_planner():
        nonlocal active_goal, planning_future, planner_status, planning_generation, planner_generation, last_status, action_queue
        if not active_goal.strip() or world.done or agent_paused:
            return
        if planning_future is not None and not planning_future.done():
            return

        if complete_active_goal('goal satisfied'):
            return

        local_actions, local_reason = local_macro_plan(active_goal, world, belief_map, belief_pose)
        if local_actions:
            action_queue.extend(local_actions)
            planner_status = local_reason
            last_status = f'{local_reason}: {len(local_actions)} actions'
            return
        if local_reason == 'search exhausted':
            action_queue.clear()
            planner_status = 'search exhausted'
            last_status = 'Search exhausted: target not found in explored map'
            active_goal = ''
            return

        planner_generation += 1
        planning_generation = planner_generation
        prompt = build_llm_prompt(active_goal, world, belief_map, belief_pose, last_scan)
        planning_future = planner_executor.submit(call_llm_planner, prompt)
        planner_status = 'asking API'
        last_status = f'planning with {OPENROUTER_MODEL}'

    def poll_planner():
        nonlocal planning_future, planner_status, action_queue, last_status, active_goal, last_llm_reply, replan_delay
        if planning_future is None or not planning_future.done():
            return

        # If a newer task was submitted while the API was thinking, ignore the old result.
        try:
            result = planning_future.result()
        except Exception as exc:
            result = {'actions': [], 'done': False, 'raw': '', 'error': str(exc)}

        planning_future = None
        if planning_generation != planner_generation:
            return

        last_llm_reply = result.get('raw', '')

        if result.get('error'):
            planner_status = 'API error'
            last_status = f"API error: {str(result['error'])[:58]}"
            replan_delay = FPS * 2
            return

        if complete_active_goal('goal satisfied'):
            return

        if result.get('done'):
            if complete_active_goal('LLM marked task complete'):
                return
            planner_status = 'LLM done rejected'
            last_status = 'LLM said done, verifier disagrees'
            replan_delay = FPS
            return

        actions = result.get('actions', [])
        if actions:
            action_queue.extend(actions)
            planner_status = 'executing API plan'
            last_status = f'LLM actions: {actions}'
            replan_delay = 0
        else:
            planner_status = 'no API action'
            last_status = 'LLM returned no usable action'
            replan_delay = FPS

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                shutdown_and_exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    shutdown_and_exit()
                elif event.key == pygame.K_TAB:
                    input_active = not input_active
                elif event.key == pygame.K_p and not input_active and active_goal:
                    agent_paused = not agent_paused
                    planner_status = 'paused' if agent_paused else 'running'
                    last_status = 'agent paused' if agent_paused else 'agent resumed'
                elif event.key == pygame.K_c and not input_active:
                    clear_agent('agent stopped')
                elif event.key == pygame.K_m and not input_active:
                    show_truth_debug = not show_truth_debug
                    last_status = f"truth minimap {'shown' if show_truth_debug else 'hidden'}"
                elif input_active:
                    if event.key == pygame.K_RETURN:

                        cmd = input_text.strip().lower()
                        submitted = input_text.strip()
                        input_text = ''
                        if cmd == 'reset':
                            belief_pose = Pose(world.spawn[0] + 0.5, world.spawn[1] + 0.5, 0.0)
                            belief_map = BeliefMap()
                            last_scan = update_belief_map_from_scan(world, belief_map, belief_pose)
                            action_queue.clear()
                            active_goal = ''
                            planner_status = 'reset'
                            last_status = 'belief reset'
                        elif cmd == 'stats':
                            unknown, free, walls, objs = belief_map.count_known()
                            last_status = f'unknown={unknown} free={free} walls={walls} objects={objs}'
                        elif cmd in {'stop', 'cancel', 'clear'}:
                            clear_agent('agent stopped')
                        elif submitted:
                            active_goal = submitted
                            action_queue.clear()
                            agent_paused = False
                            replan_delay = 0
                            planner_status = 'queued task'
                            last_status = f'task: {submitted}'
                            input_active = False
                            kick_off_planner()
                        else:
                            last_status = 'enter a task or command'
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        if event.unicode and event.unicode.isprintable():
                            input_text += event.unicode

        keys = pygame.key.get_pressed()
        poll_planner()
        complete_active_goal('goal satisfied')

        if not input_active and not world.done and active_goal and not agent_paused:
            if autonomy_action_delay > 0:
                autonomy_action_delay -= 1
            elif action_queue:
                if complete_active_goal('goal satisfied'):
                    pass
                elif should_interrupt_for_better_task_plan(active_goal, planner_status, world, belief_map):
                    action_queue.clear()
                    planner_status = 'target found; replanning'
                    last_status = 'target found; replanning efficient route'
                    kick_off_planner()
                else:
                    apply_action(action_queue.pop(0))
                    autonomy_action_delay = AUTONOMY_ACTION_DELAY
            elif planning_future is None:
                if replan_delay > 0:
                    replan_delay -= 1
                else:
                    kick_off_planner()

            space_was_down = keys[pygame.K_SPACE]

        elif not input_active and not world.done:
            # Manual control stays available so the SLAM layer can be tested without the API.
            if action_repeat_delay <= 0:
                action = move_action_from_keys(keys)
                if action is not None:
                    apply_action(action)
                    action_repeat_delay = 3
            else:
                action_repeat_delay -= 1

            space_down = keys[pygame.K_SPACE]
            if space_down and not space_was_down:
                apply_action('interact')
            space_was_down = space_down
        else:
            space_was_down = keys[pygame.K_SPACE]

        render_3d(screen, world)
        if show_truth_debug:
            draw_minimap(screen, world, belief_map, belief_pose, scale=DEBUG_MINIMAP_SCALE)
        render_explored_map(
            screen,
            belief_map,
            belief_pose,
            active_goal if active_goal else 'Type a task for API control',
            input_text,
            input_active,
            last_status,
            show_truth_debug,
            world.spawn,
            planner_status,
            len(action_queue),
            OPENROUTER_MODEL,
        )

        obs = world.get_observation()
        status_line = (
            f"Belief: ({belief_pose.x:.2f}, {belief_pose.y:.2f}) cell [{cell_of(belief_pose.x)}, {cell_of(belief_pose.y)}] | "
            f"Carry: {obs['inventory']} | "
            f"Delivered: {obs['delivered']}/{world.total_items} | "
            f"{world.message}"
        )
        controls_line = 'TAB command | Enter ask API | P pause | C stop | Arrows manual when idle | M truth debug | ESC quit'

        text1 = font.render(status_line, True, (255, 255, 255))
        text2 = small.render(controls_line, True, (255, 255, 255))
        screen.blit(text1, (10, SCREEN_H - 48))
        screen.blit(text2, (10, SCREEN_H - 24))

        if world.done:
            done_txt = font.render('SUCCESS', True, (255, 255, 0))
            screen.blit(done_txt, (SCREEN_W - MAP_PANEL_W - 110, 10))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == '__main__':
    main()
