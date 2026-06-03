import math
import random
import sys
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pygame


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
NUM_OBJECTS = 5
NUM_BARRIERS = 3
MAX_GENERATION_ATTEMPTS = 200

MAP_PANEL_W = 360
MAP_RADIUS = 8
PROMPT_RADIUS = 6
FONT_SIZE = 22
DEBUG_MINIMAP_SCALE = 10


OBJECTS = {
    'm': ('mug', (255, 100, 0)),
    'p': ('plate', (23, 255, 31)),
    't': ('towel', (0, 140, 255)),
    'o': ('bowl', (255, 23, 232)),
    's': ('spoon', (120, 80, 200)),
    'l': ('lamp', (255, 210, 0)),
}

# Same symbols as the main agent, just without the API loop.

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

        # A few broken-up walls make the SLAM view more interesting to test.
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
                item_name = OBJECTS[self.agent.carried_item][0]
                self.delivered += 1
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
            'remaining': remaining + (1 if self.agent.carried_item is not None else 0),
            'message': self.message,
            'done': self.done,
        }


@dataclass
class Pose:
    x: float
    y: float
    angle: float


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
    sin_a = math.sin(ray_angle)
    cos_a = math.cos(ray_angle)

    depth = 0.0
    while depth < MAX_DEPTH:
        depth += 0.02
        x = ox + cos_a * depth
        y = oy + sin_a * depth
        t = world.tile_at(int(x), int(y))
        if t == '#' or is_object_tile(t):
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

    title = big.render('Explored SLAM map', True, (255, 255, 255))
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
    info4 = font.render(f'visible ?:{visible_unknown} .:{visible_free} #:{visible_walls} obj:{visible_objects}', True, (210, 215, 225))
    screen.blit(info1, (start_x, y)); y += 20
    screen.blit(info2, (start_x, y)); y += 20
    screen.blit(info3, (start_x, y)); y += 20
    screen.blit(info4, (start_x, y)); y += 20

    y += 8
    status = font.render(last_status[:44], True, (255, 245, 170))
    screen.blit(status, (start_x, y)); y += 22
    goal = font.render(goal_text[:44], True, (210, 230, 255))
    screen.blit(goal, (start_x, y)); y += 22

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


def update_belief_map_from_scan(world: World, belief_map: BeliefMap, belief_pose: Pose):

    # Manual test mode uses true simulator rays but writes them into belief space.
    scan = []
    for i in range(OBS_RAYS):
        t = 0 if OBS_RAYS == 1 else i / (OBS_RAYS - 1)
        offset = -FOV / 2 + t * FOV
        depth, hit = cast_ray_truth(world, world.agent.angle + offset)
        scan.append({'angle_offset': offset, 'distance': depth, 'hit': hit})

    belief_map.update_from_scan(belief_pose, scan)


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


def main():
    pygame.init()
    pygame.display.set_caption('LLM 2D World - Simple SLAM-lite')
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, FONT_SIZE)
    small = pygame.font.SysFont(None, 18)

    map_rows = generate_simple_world(WORLD_W, WORLD_H, NUM_OBJECTS)
    world = World(map_rows)


    belief_pose = Pose(world.spawn[0] + 0.5, world.spawn[1] + 0.5, 0.0)
    belief_map = BeliefMap()


    update_belief_map_from_scan(world, belief_map, belief_pose)

    input_active = False
    input_text = ''
    last_status = 'Walk around to explore the map. TAB toggles text input.'
    action_repeat_delay = 0
    show_truth_debug = False
    space_was_down = False

    def apply_manual_action(action: str):
        nonlocal last_status
        before = Pose(world.agent.x, world.agent.y, world.agent.angle)
        before_carried = world.agent.carried_item

        world.step(action)
        update_belief_pose_from_odometry(belief_pose, before, world)

        if action == 'interact' and before_carried is None and world.agent.carried_item is not None:
            belief_map.clear_object_at(cell_of(belief_pose.x), cell_of(belief_pose.y))

        update_belief_map_from_scan(world, belief_map, belief_pose)
        last_status = world.message

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_TAB:
                    input_active = not input_active
                elif event.key == pygame.K_m and not input_active:
                    show_truth_debug = not show_truth_debug
                    last_status = f"truth minimap {'shown' if show_truth_debug else 'hidden'}"
                elif input_active:
                    if event.key == pygame.K_RETURN:

                        cmd = input_text.strip().lower()
                        input_text = ''
                        if cmd == 'reset':
                            belief_pose = Pose(world.spawn[0] + 0.5, world.spawn[1] + 0.5, 0.0)
                            belief_map = BeliefMap()
                            update_belief_map_from_scan(world, belief_map, belief_pose)
                            last_status = 'belief reset'
                        elif cmd == 'stats':
                            unknown, free, walls, objs = belief_map.count_known()
                            last_status = f'unknown={unknown} free={free} walls={walls} objects={objs}'
                        else:
                            last_status = f'unknown command: {cmd}'
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        if event.unicode and event.unicode.isprintable():
                            input_text += event.unicode

        keys = pygame.key.get_pressed()
        if not input_active and not world.done:
            if action_repeat_delay <= 0:
                action = move_action_from_keys(keys)
                if action is not None:
                    apply_manual_action(action)
                    action_repeat_delay = 3
            else:
                action_repeat_delay -= 1

            space_down = keys[pygame.K_SPACE]
            if space_down and not space_was_down:
                apply_manual_action('interact')
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
            'Manual exploration: rays update the occupancy map',
            input_text,
            input_active,
            last_status,
            show_truth_debug,
            world.spawn,
        )

        obs = world.get_observation()
        status_line = (
            f"Belief: ({belief_pose.x:.2f}, {belief_pose.y:.2f}) cell [{cell_of(belief_pose.x)}, {cell_of(belief_pose.y)}] | "
            f"Carry: {obs['inventory']} | "
            f"Delivered: {obs['delivered']}/{world.total_items} | "
            f"{world.message}"
        )
        controls_line = 'Arrows move/turn | SPACE interact | TAB reset/stats | M truth debug | ESC quit'

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
