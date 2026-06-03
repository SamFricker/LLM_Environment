import ast
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pygame

try:
    from chat import OPENROUTER_MODEL, OPENROUTER_MODELS, client as LLM_CLIENT
    LLM_IMPORT_ERROR = ''
except Exception as exc:
    OPENROUTER_MODEL = 'unavailable'
    OPENROUTER_MODELS = []
    LLM_CLIENT = None
    LLM_IMPORT_ERROR = str(exc)


SCREEN_W, SCREEN_H = 1200, 720
FPS = 60
PANEL_W = 430
GRID_SIZE = 38
VIEW_X = 24
VIEW_Y = 38
VIEW_W = SCREEN_W - PANEL_W - (VIEW_X * 2)
VIEW_H = 210
MAP_Y = VIEW_Y + VIEW_H + 30
RAYCAST_RAYS = 160
RAYCAST_FOV = math.radians(68)
RAYCAST_MAX_DEPTH = 14.0
ACTION_DELAY_FRAMES = 7
LLM_TIMEOUT_SECONDS = 20
MAX_ACTIONS = 40
MAX_LLM_TURNS_PER_TASK = 100
LOG_PATH = 'agent_run_log.jsonl'
WORLD_W = 15
WORLD_H = 9
RANDOM_WALLS = 9
SLAM_CELL = 10
SLAM_X = VIEW_X + WORLD_W * GRID_SIZE + 22
SLAM_Y = MAP_Y

ALLOWED_ACTIONS = {
    'move_north',
    'move_south',
    'move_east',
    'move_west',
    'move_forward',
    'move_back',
    'turn_left',
    'turn_right',
    'interact',
    'wait',
}

OBJECTS = {
    'm': ('mug', (255, 115, 30)),
    'p': ('plate', (55, 230, 70)),
    't': ('towel', (30, 145, 255)),
    'o': ('bowl', (255, 65, 220)),
    's': ('spoon', (150, 105, 230)),
    'l': ('lamp', (255, 215, 50)),
}

COLOR_OBJECTS = {
    'orange': 'mug',
    'green': 'plate',
    'lime': 'plate',
    'blue': 'towel',
    'cyan': 'towel',
    'azure': 'towel',
    'pink': 'bowl',
    'magenta': 'bowl',
    'purple': 'spoon',
    'violet': 'spoon',
    'yellow': 'lamp',
    'gold': 'lamp',
}

DIRECTIONS = ['east', 'south', 'west', 'north']
DIR_DELTA = {
    'east': (1, 0),
    'south': (0, 1),
    'west': (-1, 0),
    'north': (0, -1),
}

SYSTEM_PROMPT = (
    "You are a movement-command translator for a tiny gridworld SLAM demo. "
    "The user gives a natural-language command. You inspect the observation and return primitive actions. "
    "There is no hidden planner: your JSON actions are exactly what the simulator will execute. "
    "You do not receive the true world map. You only receive a SLAM map built from what the robot has seen in its raycast field of view. "
    "In that map, ? means unknown/unseen. Do not invent object coordinates or wall positions that are still ?. "
    "Return ONLY JSON: {\"rationale\":\"short reason\",\"planned_room_cells\":[[1,2],[2,2]],\"actions\":[{\"action\":\"move_east\",\"repeat\":2}]}. "
    "Allowed actions are move_north, move_south, move_east, move_west, move_forward, move_back, "
    "turn_left, turn_right, interact, wait. Absolute moves move one grid cell. "
    "move_forward and move_back use the robot heading. turn_left and turn_right rotate 90 degrees. "
    "interact picks up an object on the current cell, or drops a carried object at B/bin. "
    "Coordinates use room_cell values: zero-based interior room coordinates, with the outer wall border not counted. "
    "You are responsible for pathfinding using only discovered SLAM cells. Plan routes through adjacent known non-wall cells in room_map. "
    "Treat # as an impassable known wall. Never choose a move action whose move_affordances entry says blocked=true. "
    "If the target object/bin is not yet in discovered_objects or known_bin_room_cell, explore first. "
    "For find/search/locate commands, the task is not complete until the target appears in discovered_objects. "
    "Good exploration means turn to scan, then move one cell at a time toward adjacent ? frontier cells when that move is not a known wall. "
    "Aim to uncover as much unknown space as possible: prefer frontier cells and routes that reveal new ? cells, using frontier_room_cells and unknown_room_cells. "
    "Do not return long repeated moves through unknown territory; take one or two safe exploratory steps, then you will receive a fresh SLAM map and can retry. "
    "If a known route to the target exists, use it. If no known route exists and unknown cells remain, keep exploring instead of waiting or declaring failure. "
    "wall_room_cells contains only discovered walls; never include those known wall coordinates in planned_room_cells. "
    "Each planned_room_cells entry should be a known floor/object/bin cell, or a single adjacent ? cell when deliberately exploring. "
    "Do not repeat a movement direction after last_message says it was blocked; turn or choose a different frontier. "
    "Prefer absolute moves (move_north/south/east/west) for grid pathfinding; use turn/move_forward only when the user asks for heading-based motion. "
    "If the target object is on the current cell, use interact immediately. "
    "To pick up an object, first move onto that object's room_cell, then use interact. "
    "If carrying an object and asked to drop/deliver/bin it, move to B/bin, then use interact. "
    "Colour/color words are object references. Resolve 'blue', 'blue object', 'blue item', or 'blue thing' as towel. "
    "Colour mapping: orange=mug, green/lime=plate, blue/cyan/azure=towel, pink/magenta=bowl, "
    "purple/violet=spoon, yellow/gold=lamp. "
    f"Use no more than {MAX_ACTIONS} primitive actions. If the command is ambiguous, choose a small safe action."
)


@dataclass
class Agent:
    x: int
    y: int
    heading: int = 0
    carried: Optional[str] = None

    @property
    def heading_name(self) -> str:
        return DIRECTIONS[self.heading % 4]


def room_cell(cell: Tuple[int, int]) -> List[int]:
    return [cell[0] - 1, cell[1] - 1]


def reachable_cells(grid: List[List[str]], start: Tuple[int, int]) -> List[Tuple[int, int]]:
    queue = [start]
    seen = {start}
    while queue:
        x, y = queue.pop(0)
        for dx, dy in DIR_DELTA.values():
            nx, ny = x + dx, y + dy
            if (nx, ny) in seen:
                continue
            if grid[ny][nx] == '#':
                continue
            seen.add((nx, ny))
            queue.append((nx, ny))
    return list(seen)


def generate_random_world() -> List[List[str]]:
    object_symbols = list(OBJECTS.keys())

    for _attempt in range(200):
        grid = [['.' for _x in range(WORLD_W)] for _y in range(WORLD_H)]
        for x in range(WORLD_W):
            grid[0][x] = '#'
            grid[WORLD_H - 1][x] = '#'
        for y in range(WORLD_H):
            grid[y][0] = '#'
            grid[y][WORLD_W - 1] = '#'

        interior = [(x, y) for y in range(1, WORLD_H - 1) for x in range(1, WORLD_W - 1)]
        for x, y in random.sample(interior, min(RANDOM_WALLS, len(interior))):
            grid[y][x] = '#'

        floors = [(x, y) for x, y in interior if grid[y][x] == '.']
        if len(floors) < len(object_symbols) + 2:
            continue

        agent_cell = random.choice(floors)
        reachable = reachable_cells(grid, agent_cell)
        if len(reachable) < len(object_symbols) + 2:
            continue

        placement_pool = [cell for cell in reachable if cell != agent_cell]
        if len(placement_pool) < len(object_symbols) + 1:
            continue

        chosen = random.sample(placement_pool, len(object_symbols) + 1)
        bin_cell = chosen[0]
        object_cells = chosen[1:]

        ax, ay = agent_cell
        bx, by = bin_cell
        grid[ay][ax] = 'A'
        grid[by][bx] = 'B'
        for symbol, (x, y) in zip(object_symbols, object_cells):
            grid[y][x] = symbol
        return grid

    raise RuntimeError('Failed to generate a reachable random world.')


class World:
    def __init__(self):
        self.reset()

    def reset(self):
        self.grid = generate_random_world()
        self.bin_cell = (0, 0)
        self.spawn_cell = (0, 0)
        self.delivered: List[str] = []
        self.message = 'Ready. Type a movement command and press Enter.'
        self.last_blocked = False

        for y, row in enumerate(self.grid):
            for x, ch in enumerate(row):
                if ch == 'A':
                    self.agent = Agent(x, y, 0)
                    self.spawn_cell = (x, y)
                    self.grid[y][x] = '.'
                elif ch == 'B':
                    self.bin_cell = (x, y)
                    self.grid[y][x] = '.'

        self.slam_grid = [['?' for _x in range(self.width())] for _y in range(self.height())]
        self.update_slam_from_fov()

    def width(self) -> int:
        return len(self.grid[0])

    def height(self) -> int:
        return len(self.grid)

    def tile_at(self, x: int, y: int) -> str:
        if x < 0 or y < 0 or y >= self.height() or x >= self.width():
            return '#'
        return self.grid[y][x]

    def is_wall(self, x: int, y: int) -> bool:
        return self.tile_at(x, y) == '#'

    def visible_symbol_at(self, x: int, y: int) -> str:
        if (x, y) == self.bin_cell:
            return 'B'
        return self.tile_at(x, y)

    def slam_symbol_at(self, x: int, y: int) -> str:
        if x < 0 or y < 0 or y >= self.height() or x >= self.width():
            return '#'
        return self.slam_grid[y][x]

    def mark_slam_cell(self, x: int, y: int) -> str:
        if x < 0 or y < 0 or y >= self.height() or x >= self.width():
            return '#'
        symbol = self.visible_symbol_at(x, y)
        self.slam_grid[y][x] = symbol
        return symbol

    def update_slam_from_fov(self):
        if not hasattr(self, 'slam_grid'):
            return

        self.mark_slam_cell(self.agent.x, self.agent.y)
        heading = heading_angle(self.agent.heading_name)
        ray_angle = heading - RAYCAST_FOV / 2
        angle_step = RAYCAST_FOV / max(1, RAYCAST_RAYS - 1)
        step = 0.045

        for _ray in range(RAYCAST_RAYS):
            depth = 0.0
            while depth <= RAYCAST_MAX_DEPTH:
                sx = self.agent.x + 0.5 + math.cos(ray_angle) * depth
                sy = self.agent.y + 0.5 + math.sin(ray_angle) * depth
                mx = int(sx)
                my = int(sy)
                symbol = self.mark_slam_cell(mx, my)

                if symbol == '#' or symbol == 'B' or symbol in OBJECTS:
                    break
                depth += step
            ray_angle += angle_step

    def at_bin(self) -> bool:
        return (self.agent.x, self.agent.y) == self.bin_cell

    def object_count(self) -> int:
        return sum(1 for row in self.grid for ch in row if ch in OBJECTS)

    def move_delta(self, dx: int, dy: int, label: str):
        nx = self.agent.x + dx
        ny = self.agent.y + dy
        if self.is_wall(nx, ny):
            self.last_blocked = True
            rx, ry = room_cell((nx, ny))
            self.message = f'{label} blocked by wall at room cell ({rx}, {ry}).'
            return

        self.last_blocked = False
        self.agent.x = nx
        self.agent.y = ny
        rx, ry = room_cell((nx, ny))
        self.message = f'{label} to room cell ({rx}, {ry}).'

    def step(self, action: str):
        action = action.strip().lower()

        if action == 'turn_left':
            self.agent.heading = (self.agent.heading - 1) % 4
            self.last_blocked = False
            self.message = f'Turned left. Heading {self.agent.heading_name}.'
        elif action == 'turn_right':
            self.agent.heading = (self.agent.heading + 1) % 4
            self.last_blocked = False
            self.message = f'Turned right. Heading {self.agent.heading_name}.'
        elif action == 'move_forward':
            dx, dy = DIR_DELTA[self.agent.heading_name]
            self.move_delta(dx, dy, 'Moved forward')
        elif action == 'move_back':
            dx, dy = DIR_DELTA[self.agent.heading_name]
            self.move_delta(-dx, -dy, 'Moved back')
        elif action == 'move_north':
            self.agent.heading = DIRECTIONS.index('north')
            self.move_delta(0, -1, 'Moved north')
        elif action == 'move_south':
            self.agent.heading = DIRECTIONS.index('south')
            self.move_delta(0, 1, 'Moved south')
        elif action == 'move_east':
            self.agent.heading = DIRECTIONS.index('east')
            self.move_delta(1, 0, 'Moved east')
        elif action == 'move_west':
            self.agent.heading = DIRECTIONS.index('west')
            self.move_delta(-1, 0, 'Moved west')
        elif action == 'interact':
            self.interact()
        elif action == 'wait':
            self.last_blocked = False
            self.message = 'Waited.'
        else:
            self.last_blocked = False
            self.message = f'Ignored unsupported action: {action}'

    def interact(self):
        x, y = self.agent.x, self.agent.y
        tile = self.tile_at(x, y)

        if self.agent.carried is not None:
            if self.at_bin():
                name = OBJECTS[self.agent.carried][0]
                self.delivered.append(self.agent.carried)
                self.agent.carried = None
                self.message = f'Dropped {name} in the bin.'
            else:
                self.message = 'Not at the bin, so the carried item stays with the robot.'
            return

        if tile in OBJECTS:
            self.agent.carried = tile
            self.grid[y][x] = '.'
            self.message = f'Picked up {OBJECTS[tile][0]}.'
            return

        self.message = 'Nothing to interact with here.'

    def display_rows(self) -> List[str]:
        rows = []
        for y, row in enumerate(self.grid):
            chars = []
            for x, ch in enumerate(row):
                if (x, y) == (self.agent.x, self.agent.y):
                    chars.append('@')
                elif (x, y) == self.bin_cell:
                    chars.append('B')
                else:
                    chars.append(ch)
            rows.append(''.join(chars))
        return rows

    def slam_display_rows(self) -> List[str]:
        rows = []
        for y in range(self.height()):
            chars = []
            for x in range(self.width()):
                if (x, y) == (self.agent.x, self.agent.y):
                    chars.append('@')
                else:
                    chars.append(self.slam_symbol_at(x, y))
            rows.append(''.join(chars))
        return rows

    def room_rows(self) -> List[str]:
        rows = []
        for y in range(1, self.height() - 1):
            chars = []
            for x in range(1, self.width() - 1):
                if (x, y) == (self.agent.x, self.agent.y):
                    chars.append('@')
                elif (x, y) == self.bin_cell:
                    chars.append('B')
                else:
                    chars.append(self.tile_at(x, y))
            rows.append(''.join(chars))
        return rows

    def slam_room_rows(self) -> List[str]:
        rows = []
        for y in range(1, self.height() - 1):
            chars = []
            for x in range(1, self.width() - 1):
                if (x, y) == (self.agent.x, self.agent.y):
                    chars.append('@')
                else:
                    chars.append(self.slam_symbol_at(x, y))
            rows.append(''.join(chars))
        return rows

    def indexed_room_map(self) -> Dict[str, Any]:
        rows = self.room_rows()
        width = len(rows[0]) if rows else 0
        return {
            'x_columns': list(range(width)),
            'rows': [
                {'y': y, 'cells': row}
                for y, row in enumerate(rows)
            ],
        }

    def slam_indexed_room_map(self) -> Dict[str, Any]:
        rows = self.slam_room_rows()
        width = len(rows[0]) if rows else 0
        return {
            'x_columns': list(range(width)),
            'rows': [
                {'y': y, 'cells': row}
                for y, row in enumerate(rows)
            ],
        }

    def room_bounds(self) -> Dict[str, int]:
        return {
            'min_x': 0,
            'max_x': self.width() - 3,
            'min_y': 0,
            'max_y': self.height() - 3,
        }

    def wall_room_cells(self) -> List[List[int]]:
        walls = []
        for y in range(1, self.height() - 1):
            for x in range(1, self.width() - 1):
                if self.tile_at(x, y) == '#':
                    walls.append(room_cell((x, y)))
        return walls

    def slam_wall_room_cells(self) -> List[List[int]]:
        walls = []
        for y in range(1, self.height() - 1):
            for x in range(1, self.width() - 1):
                if self.slam_symbol_at(x, y) == '#':
                    walls.append(room_cell((x, y)))
        return walls

    def traversable_room_cells(self) -> List[List[int]]:
        cells = []
        for y in range(1, self.height() - 1):
            for x in range(1, self.width() - 1):
                if self.tile_at(x, y) != '#':
                    cells.append(room_cell((x, y)))
        return cells

    def slam_traversable_room_cells(self) -> List[List[int]]:
        cells = []
        for y in range(1, self.height() - 1):
            for x in range(1, self.width() - 1):
                symbol = self.slam_symbol_at(x, y)
                if symbol != '#' and symbol != '?':
                    cells.append(room_cell((x, y)))
        return cells

    def nearby(self, radius: int = 2) -> List[str]:
        rows = []
        for y in range(self.agent.y - radius, self.agent.y + radius + 1):
            chars = []
            for x in range(self.agent.x - radius, self.agent.x + radius + 1):
                if (x, y) == (self.agent.x, self.agent.y):
                    chars.append('@')
                elif (x, y) == self.bin_cell:
                    chars.append('B')
                else:
                    chars.append(self.tile_at(x, y))
            rows.append(''.join(chars))
        return rows

    def slam_nearby(self, radius: int = 2) -> List[str]:
        rows = []
        for y in range(self.agent.y - radius, self.agent.y + radius + 1):
            chars = []
            for x in range(self.agent.x - radius, self.agent.x + radius + 1):
                if (x, y) == (self.agent.x, self.agent.y):
                    chars.append('@')
                else:
                    chars.append(self.slam_symbol_at(x, y))
            rows.append(''.join(chars))
        return rows

    def slam_known_objects(self) -> List[Dict[str, Any]]:
        objects = []
        for y, row in enumerate(self.slam_grid):
            for x, ch in enumerate(row):
                if ch in OBJECTS:
                    objects.append({
                        'symbol': ch,
                        'name': OBJECTS[ch][0],
                        'room_cell': room_cell((x, y)),
                    })
        return objects

    def slam_unknown_room_cells(self) -> List[List[int]]:
        unknown = []
        for y in range(1, self.height() - 1):
            for x in range(1, self.width() - 1):
                if self.slam_symbol_at(x, y) == '?':
                    unknown.append(room_cell((x, y)))
        return unknown

    def slam_frontier_room_cells(self) -> List[List[int]]:
        frontiers = []
        for y in range(1, self.height() - 1):
            for x in range(1, self.width() - 1):
                if self.slam_symbol_at(x, y) != '?':
                    continue
                for dx, dy in DIR_DELTA.values():
                    if self.slam_symbol_at(x + dx, y + dy) not in {'?', '#'}:
                        frontiers.append(room_cell((x, y)))
                        break
        return frontiers

    def move_affordances(self) -> Dict[str, Dict[str, Any]]:
        action_deltas = {
            'move_north': (0, -1),
            'move_south': (0, 1),
            'move_east': (1, 0),
            'move_west': (-1, 0),
        }
        forward_dx, forward_dy = DIR_DELTA[self.agent.heading_name]
        action_deltas['move_forward'] = (forward_dx, forward_dy)
        action_deltas['move_back'] = (-forward_dx, -forward_dy)

        affordances = {}
        for action, (dx, dy) in action_deltas.items():
            tx = self.agent.x + dx
            ty = self.agent.y + dy
            tile = self.tile_at(tx, ty)
            entry: Dict[str, Any] = {
                'target_room_cell': room_cell((tx, ty)),
                'target_symbol': 'B' if (tx, ty) == self.bin_cell else tile,
                'blocked': tile == '#',
            }
            if tile in OBJECTS:
                entry['target_object'] = OBJECTS[tile][0]
            affordances[action] = entry
        return affordances

    def slam_move_affordances(self) -> Dict[str, Dict[str, Any]]:
        action_deltas = {
            'move_north': (0, -1),
            'move_south': (0, 1),
            'move_east': (1, 0),
            'move_west': (-1, 0),
        }
        forward_dx, forward_dy = DIR_DELTA[self.agent.heading_name]
        action_deltas['move_forward'] = (forward_dx, forward_dy)
        action_deltas['move_back'] = (-forward_dx, -forward_dy)

        affordances = {}
        for action, (dx, dy) in action_deltas.items():
            tx = self.agent.x + dx
            ty = self.agent.y + dy
            symbol = self.slam_symbol_at(tx, ty)
            entry: Dict[str, Any] = {
                'target_room_cell': room_cell((tx, ty)),
                'target_symbol': symbol,
                'known': symbol != '?',
                'unknown': symbol == '?',
                'blocked': symbol == '#',
            }
            if symbol in OBJECTS:
                entry['target_object'] = OBJECTS[symbol][0]
            affordances[action] = entry
        return affordances

    def directional_clearance(self) -> Dict[str, Dict[str, Any]]:
        clearance = {}
        for direction, (dx, dy) in DIR_DELTA.items():
            steps = 0
            seen: List[List[int]] = []
            x = self.agent.x + dx
            y = self.agent.y + dy

            while not self.is_wall(x, y):
                seen.append(room_cell((x, y)))
                steps += 1
                x += dx
                y += dy

            clearance[direction] = {
                'move_action': f'move_{direction}',
                'clear_steps_before_wall': steps,
                'clear_room_cells_before_wall': seen,
                'first_wall_room_cell': room_cell((x, y)),
            }
        return clearance

    def slam_directional_clearance(self) -> Dict[str, Dict[str, Any]]:
        clearance = {}
        for direction, (dx, dy) in DIR_DELTA.items():
            steps = 0
            seen: List[List[int]] = []
            first_unknown = None
            first_wall = None
            x = self.agent.x + dx
            y = self.agent.y + dy

            while True:
                symbol = self.slam_symbol_at(x, y)
                if symbol == '#':
                    first_wall = room_cell((x, y))
                    break
                if symbol == '?':
                    first_unknown = room_cell((x, y))
                    break
                seen.append(room_cell((x, y)))
                steps += 1
                x += dx
                y += dy

            clearance[direction] = {
                'move_action': f'move_{direction}',
                'known_clear_steps_before_wall_or_unknown': steps,
                'known_clear_room_cells': seen,
                'first_unknown_room_cell': first_unknown,
                'first_known_wall_room_cell': first_wall,
            }
        return clearance

    def observation(self, command: str) -> Dict[str, Any]:
        carried = OBJECTS[self.agent.carried][0] if self.agent.carried in OBJECTS else None
        current_tile = self.tile_at(self.agent.x, self.agent.y)
        current_object = OBJECTS[current_tile][0] if current_tile in OBJECTS else None
        known_bin_room_cell = None
        if self.slam_symbol_at(*self.bin_cell) == 'B':
            known_bin_room_cell = room_cell(self.bin_cell)

        return {
            'user_command': command,
            'coordinate_system': 'All room_cell coordinates are zero-based interior coordinates. The outer wall border is not counted.',
            'map_visibility': 'SLAM-only. ? cells are unknown/unseen. The true map is not provided to the LLM.',
            'room_size': {
                'width': self.width() - 2,
                'height': self.height() - 2,
            },
            'room_bounds': self.room_bounds(),
            'spawn_room_cell': room_cell(self.spawn_cell),
            'legend': {
                '#': 'wall',
                '.': 'floor',
                '?': 'unknown/unseen',
                '@': 'robot',
                'B': 'bin/drop location',
                **{symbol: name for symbol, (name, _color) in OBJECTS.items()},
            },
            'color_object_aliases': COLOR_OBJECTS,
            'robot': {
                'room_cell': room_cell((self.agent.x, self.agent.y)),
                'heading': self.agent.heading_name,
                'inventory': [carried] if carried else [],
                'current_cell_symbol': 'B' if self.at_bin() else current_tile,
                'current_cell_object': current_object,
                'can_pick_up_now': current_object is not None and carried is None,
                'can_drop_now': carried is not None and self.at_bin(),
            },
            'interaction_rules': {
                'pick_up': 'If can_pick_up_now is true and the command asks for that object, return interact.',
                'drop': 'If can_drop_now is true and the command asks to drop/deliver/bin the carried item, return interact.',
            },
            'allowed_actions': sorted(ALLOWED_ACTIONS),
            'room_map': self.slam_room_rows(),
            'room_map_indexed': self.slam_indexed_room_map(),
            'wall_room_cells': self.slam_wall_room_cells(),
            'traversable_room_cells': self.slam_traversable_room_cells(),
            'move_affordances': self.slam_move_affordances(),
            'directional_clearance': self.slam_directional_clearance(),
            'nearby_5x5': self.slam_nearby(),
            'objects': self.slam_known_objects(),
            'discovered_objects': self.slam_known_objects(),
            'unknown_room_cells': self.slam_unknown_room_cells(),
            'frontier_room_cells': self.slam_frontier_room_cells(),
            'slam_coverage': {
                'known_cells': (self.width() - 2) * (self.height() - 2) - len(self.slam_unknown_room_cells()),
                'unknown_cells': len(self.slam_unknown_room_cells()),
            },
            'bin_room_cell': known_bin_room_cell,
            'known_bin_room_cell': known_bin_room_cell,
            'delivered_items': [OBJECTS[ch][0] for ch in self.delivered],
            'last_message': self.message,
        }


def append_log(record: Dict[str, Any]):
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as log_file:
            log_file.write(json.dumps(record, ensure_ascii=True) + '\n')
    except Exception:
        pass


def build_prompt(world: World, command: str) -> str:
    obs = world.observation(command)
    return 'Observation JSON:\n' + json.dumps(obs, separators=(',', ':'))


def build_repair_prompt(world: World, command: str, previous_raw: str, previous_actions: List[str], preflight_error: str) -> str:
    payload = {
        'repair_instruction': (
            'Your previous action list was rejected because it tried to move into a known SLAM wall. '
            'Use the same user_command and fresh SLAM observation to choose a different known route. '
            'If the target or route is still unknown, explore a nearby ? frontier one step at a time instead of inventing a path.'
        ),
        'preflight_error': preflight_error,
        'previous_raw_response': previous_raw,
        'previous_actions': previous_actions,
        'fresh_observation': world.observation(command),
    }
    return 'Repair the rejected route. Return ONLY corrected JSON.\n' + json.dumps(payload, separators=(',', ':'))


def build_continue_prompt(world: World, command: str, feedback: str) -> str:
    payload = {
        'continue_instruction': (
            'The task is not complete yet. Use the fresh observation to choose the next valid action batch. '
            'Do not repeat a route that the simulator already reported as blocked. '
            'If the requested object/bin is not in the SLAM map yet, keep exploring unknown ? frontier cells, then retry until the user_command is satisfied. '
            'For find/search/locate tasks, do not stop until the target appears in discovered_objects. '
            'Prefer actions that increase slam_coverage by revealing new unknown_room_cells.'
        ),
        'feedback': feedback,
        'fresh_observation': world.observation(command),
    }
    return 'Continue the task. Return ONLY corrected JSON.\n' + json.dumps(payload, separators=(',', ':'))


def command_target_symbol(command: str) -> Optional[str]:
    text = command.lower().replace('-', ' ')
    words = set(text.split())

    for color, object_name in COLOR_OBJECTS.items():
        if color in words:
            for symbol, (name, _color) in OBJECTS.items():
                if name == object_name:
                    return symbol

    for symbol, (name, _color) in OBJECTS.items():
        if symbol in words or name in words:
            return symbol

    return None


def command_mentions_bin(command: str) -> bool:
    words = set(command.lower().replace('-', ' ').split())
    return bool(words & {'bin', 'drop', 'deliver', 'bring', 'return'})


def command_wants_pickup(command: str) -> bool:
    text = command.lower()
    words = set(text.replace('-', ' ').split())
    return 'pick up' in text or bool(words & {'pickup', 'pick', 'grab', 'take', 'collect'})


def command_wants_navigation(command: str) -> bool:
    words = set(command.lower().replace('-', ' ').split())
    return bool(words & {'go', 'move', 'walk', 'navigate', 'reach', 'to'})


def command_wants_find(command: str) -> bool:
    words = set(command.lower().replace('-', ' ').split())
    return bool(words & {'find', 'search', 'locate', 'seek'})


def task_complete(world: World, command: str) -> Tuple[bool, str]:
    target = command_target_symbol(command)

    if target is not None:
        target_name = OBJECTS[target][0]
        if command_mentions_bin(command):
            if target in world.delivered:
                return True, f'{target_name} delivered to bin'
            return False, f'{target_name} not delivered yet'

        if command_wants_pickup(command):
            if world.agent.carried == target:
                return True, f'{target_name} picked up'
            return False, f'{target_name} not picked up yet'

        if command_wants_find(command):
            if world.agent.carried == target:
                return True, f'{target_name} found and picked up'
            for item in world.slam_known_objects():
                if item['symbol'] == target:
                    return True, f'{target_name} found in SLAM map at {item["room_cell"]}'
            return False, f'{target_name} not found in SLAM map yet; continue exploring unknown frontiers'

        if command_wants_navigation(command):
            if world.tile_at(world.agent.x, world.agent.y) == target:
                return True, f'robot reached {target_name}'
            if world.agent.carried == target:
                return True, f'robot reached and picked up {target_name}'
            return False, f'robot has not reached {target_name} yet'

    if command_mentions_bin(command) and target is None:
        if world.at_bin():
            return True, 'robot reached bin'
        return False, 'robot has not reached bin yet'

    return False, 'no explicit completion condition; execute one valid batch'


def parse_jsonish(text: str) -> Optional[Any]:
    stripped = text.strip()
    if stripped.startswith('```'):
        stripped = stripped.strip('`').strip()
        if stripped.lower().startswith('json'):
            stripped = stripped[4:].strip()

    candidates = [stripped]
    first = stripped.find('{')
    last = stripped.rfind('}')
    if first != -1 and last > first:
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


def add_action(actions: List[str], action: Any, repeat: Any = 1):
    if not isinstance(action, str):
        return
    action = action.strip().lower()
    if action not in ALLOWED_ACTIONS:
        return
    try:
        repeat = int(repeat)
    except Exception:
        repeat = 1
    repeat = max(1, min(repeat, MAX_ACTIONS))
    for _ in range(repeat):
        if len(actions) >= MAX_ACTIONS:
            return
        actions.append(action)


def parse_actions(raw: str) -> Tuple[List[str], str]:
    payload = parse_jsonish(raw)
    if payload is None:
        return [], 'Could not parse JSON.'

    actions: List[str] = []
    rationale = ''

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                add_action(actions, item)
            elif isinstance(item, dict):
                add_action(actions, item.get('action'), item.get('repeat', 1))
        return actions, 'Bare action list.'

    if not isinstance(payload, dict):
        return [], 'Parsed JSON was not an object.'

    rationale = str(payload.get('rationale') or payload.get('reason') or '').strip()
    if isinstance(payload.get('actions'), list):
        for item in payload['actions']:
            if isinstance(item, str):
                add_action(actions, item)
            elif isinstance(item, dict):
                add_action(actions, item.get('action'), item.get('repeat', 1))
    elif isinstance(payload.get('action'), str):
        add_action(actions, payload.get('action'), payload.get('repeat', 1))

    return actions, rationale


def preflight_actions(world: World, actions: List[str]) -> Tuple[List[str], str]:
    x = world.agent.x
    y = world.agent.y
    heading = world.agent.heading
    valid: List[str] = []

    for index, action in enumerate(actions, start=1):
        action = action.strip().lower()

        if action == 'turn_left':
            heading = (heading - 1) % 4
            valid.append(action)
            continue
        if action == 'turn_right':
            heading = (heading + 1) % 4
            valid.append(action)
            continue
        if action in {'interact', 'wait'}:
            valid.append(action)
            continue

        if action == 'move_forward':
            dx, dy = DIR_DELTA[DIRECTIONS[heading % 4]]
        elif action == 'move_back':
            dx, dy = DIR_DELTA[DIRECTIONS[heading % 4]]
            dx, dy = -dx, -dy
        elif action == 'move_north':
            heading = DIRECTIONS.index('north')
            dx, dy = 0, -1
        elif action == 'move_south':
            heading = DIRECTIONS.index('south')
            dx, dy = 0, 1
        elif action == 'move_east':
            heading = DIRECTIONS.index('east')
            dx, dy = 1, 0
        elif action == 'move_west':
            heading = DIRECTIONS.index('west')
            dx, dy = -1, 0
        else:
            return valid, f'Preflight stopped at action {index}: unsupported action {action}.'

        nx = x + dx
        ny = y + dy
        known_symbol = world.slam_symbol_at(nx, ny)
        if known_symbol == '#':
            rx, ry = room_cell((nx, ny))
            return valid, f'Preflight stopped before action {index} ({action}): known SLAM wall at room cell ({rx}, {ry}).'

        valid.append(action)
        if known_symbol == '?':
            return valid, ''
        x = nx
        y = ny

    return valid, ''


def number_from_words(text: str, default: int = 1) -> int:
    words = {
        'once': 1,
        'one': 1,
        'twice': 2,
        'two': 2,
        'thrice': 3,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
    }
    for token in text.lower().replace('-', ' ').split():
        if token.isdigit():
            return max(1, min(int(token), MAX_ACTIONS))
        if token in words:
            return words[token]
    return default


def offline_movement_fallback(command: str) -> Tuple[List[str], str]:
    text = command.lower()
    repeat = number_from_words(text)
    actions: List[str] = []

    if 'turn left' in text or 'rotate left' in text:
        actions.append('turn_left')
    if 'turn right' in text or 'rotate right' in text:
        actions.append('turn_right')

    if 'interact' in text or 'pick up' in text or 'pickup' in text or 'drop' in text or 'grab' in text:
        actions.append('interact')

    direction_action = None
    if 'north' in text or 'up' in text:
        direction_action = 'move_north'
    elif 'south' in text or 'down' in text:
        direction_action = 'move_south'
    elif 'east' in text:
        direction_action = 'move_east'
    elif 'west' in text:
        direction_action = 'move_west'
    elif 'forward' in text or 'ahead' in text:
        direction_action = 'move_forward'
    elif 'back' in text or 'backward' in text:
        direction_action = 'move_back'
    elif 'right' in text and 'turn right' not in text and 'rotate right' not in text:
        direction_action = 'move_east'
    elif 'left' in text and 'turn left' not in text and 'rotate left' not in text:
        direction_action = 'move_west'

    if direction_action:
        actions.extend([direction_action] * repeat)

    if not actions:
        actions = ['wait']

    actions = actions[:MAX_ACTIONS]
    return actions, 'Offline fallback used because the API/model was unavailable.'


def configured_models() -> List[str]:
    models = []
    for model in [
        OPENROUTER_MODEL,
        *list(OPENROUTER_MODELS or []),
        'meta-llama/llama-3.2-3b-instruct:free',
        'openrouter/auto',
    ]:
        model = (model or '').strip()
        if model and model != 'unavailable' and model not in models:
            models.append(model)
    return models


def ask_llm(prompt: str, command: str) -> Dict[str, Any]:
    if LLM_CLIENT is None:
        actions, rationale = offline_movement_fallback(command)
        return {
            'raw': f'OFFLINE FALLBACK: {rationale}',
            'actions': actions,
            'rationale': rationale,
            'latency': 0,
            'error': f'Missing OPENROUTER_API_KEY or client import failed. {LLM_IMPORT_ERROR}',
            'used_fallback': True,
            'model': 'offline_fallback',
        }

    errors = []
    started_all = time.perf_counter()
    for model in configured_models()[:3]:
        started = time.perf_counter()
        try:
            response = LLM_CLIENT.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': prompt},
                ],
                temperature=0,
                max_tokens=260,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            latency = round(time.perf_counter() - started, 2)
            raw = response.choices[0].message.content or ''
            actions, rationale = parse_actions(raw)
            return {
                'raw': raw,
                'actions': actions,
                'rationale': rationale,
                'latency': latency,
                'error': '',
                'used_fallback': False,
                'model': model,
            }
        except Exception as exc:
            errors.append(f'{model}: {str(exc)[:180]}')

    actions, rationale = offline_movement_fallback(command)
    return {
        'raw': 'OFFLINE FALLBACK after API errors:\n' + '\n'.join(errors),
        'actions': actions,
        'rationale': rationale,
        'latency': round(time.perf_counter() - started_all, 2),
        'error': ' | '.join(errors),
        'used_fallback': True,
        'model': 'offline_fallback',
    }


def wrap_text(text: str, font, width: int, max_lines: int) -> List[str]:
    if max_lines <= 0:
        return []

    lines: List[str] = []
    for source in (text or '').replace('\t', '    ').splitlines() or ['']:
        current = ''
        for word in source.split(' '):
            candidate = word if not current else current + ' ' + word
            if font.size(candidate)[0] <= width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
            if len(lines) >= max_lines:
                lines[-1] = lines[-1][:max(0, len(lines[-1]) - 3)] + '...'
                return lines
        lines.append(current)
        if len(lines) >= max_lines:
            lines[-1] = lines[-1][:max(0, len(lines[-1]) - 3)] + '...'
            return lines
    return lines


def tile_color(ch: str) -> Tuple[int, int, int]:
    if ch == '#':
        return (244, 244, 238)
    if ch == '.':
        return (232, 235, 226)
    if ch == 'B':
        return (74, 145, 235)
    if ch in OBJECTS:
        return OBJECTS[ch][1]
    return (160, 165, 170)


def map_tile_color(ch: str) -> Tuple[int, int, int]:
    if ch == '#':
        return (250, 250, 244)
    if ch == '.':
        return (205, 215, 199)
    return tile_color(ch)


def slam_tile_color(ch: str) -> Tuple[int, int, int]:
    if ch == '?':
        return (118, 128, 135)
    if ch == '#':
        return (250, 250, 244)
    if ch == '.':
        return (205, 215, 199)
    return tile_color(ch)


def heading_angle(heading_name: str) -> float:
    return {
        'east': 0.0,
        'south': math.pi / 2,
        'west': math.pi,
        'north': -math.pi / 2,
    }[heading_name]


def raycast_cell(world: World, angle: float) -> Tuple[float, str]:
    ox = world.agent.x + 0.5
    oy = world.agent.y + 0.5
    step = 0.025
    depth = step

    while depth <= RAYCAST_MAX_DEPTH:
        sx = ox + math.cos(angle) * depth
        sy = oy + math.sin(angle) * depth
        mx = int(sx)
        my = int(sy)

        if (mx, my) == world.bin_cell:
            return depth, 'B'

        tile = world.tile_at(mx, my)
        if tile == '#' or tile in OBJECTS:
            return depth, tile

        depth += step

    return RAYCAST_MAX_DEPTH, '.'


def shade_color(color: Tuple[int, int, int], distance: float) -> Tuple[int, int, int]:
    fog_color = (205, 216, 224)
    fog = max(0.0, min(1.0, distance / RAYCAST_MAX_DEPTH))
    light = max(0.42, 1.04 - fog * 0.48)
    fog_mix = fog * 0.7

    shaded = [max(0, min(255, int(component * light))) for component in color]
    return tuple(
        max(0, min(255, int(shaded[i] * (1.0 - fog_mix) + fog_color[i] * fog_mix)))
        for i in range(3)
    )


def lerp_color(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def draw_first_person_view(screen, world: World, font, small):
    view_rect = pygame.Rect(VIEW_X, VIEW_Y, VIEW_W, VIEW_H)
    pygame.draw.rect(screen, (211, 220, 226), view_rect)

    horizon = VIEW_Y + VIEW_H // 2
    for yy in range(VIEW_H // 2):
        t = yy / max(1, VIEW_H // 2 - 1)
        sky = lerp_color((140, 183, 218), (206, 222, 232), t)
        floor = lerp_color((178, 186, 164), (216, 219, 205), t)
        pygame.draw.line(screen, sky, (VIEW_X, VIEW_Y + yy), (VIEW_X + VIEW_W, VIEW_Y + yy))
        pygame.draw.line(screen, floor, (VIEW_X, horizon + yy), (VIEW_X + VIEW_W, horizon + yy))

    heading = heading_angle(world.agent.heading_name)
    ray_angle = heading - RAYCAST_FOV / 2
    angle_step = RAYCAST_FOV / max(1, RAYCAST_RAYS - 1)
    column_w = VIEW_W / RAYCAST_RAYS

    old_clip = screen.get_clip()
    screen.set_clip(view_rect)
    for ray in range(RAYCAST_RAYS):
        distance, tile = raycast_cell(world, ray_angle)
        corrected = max(0.001, distance * math.cos(ray_angle - heading))

        wall_h = int(min(VIEW_H * 1.25, (VIEW_H * 0.92) / corrected))
        y0 = horizon - wall_h // 2
        x0 = VIEW_X + int(ray * column_w)
        width = max(1, int(math.ceil(column_w)))

        base = tile_color(tile)
        if tile == '.':
            base = (205, 216, 224)
        color = shade_color(base, corrected)
        pygame.draw.rect(screen, color, (x0, y0, width, wall_h))

        edge = tuple(max(0, value - 24) for value in color)
        pygame.draw.line(screen, edge, (x0, y0), (x0, y0 + wall_h))
        ray_angle += angle_step
    screen.set_clip(old_clip)

    pygame.draw.rect(screen, (135, 150, 162), view_rect, 1)
    title = font.render('First-person raycast view', True, (35, 42, 50))
    screen.blit(title, (VIEW_X, 10))
    facing = small.render(f'facing {world.agent.heading_name}', True, (55, 65, 74))
    screen.blit(facing, (VIEW_X + VIEW_W - facing.get_width(), 14))


def draw_grid(screen, world: World, font, small):
    origin_x = VIEW_X
    origin_y = MAP_Y
    rows = world.display_rows()

    title = font.render('True simulation map', True, (35, 42, 50))
    screen.blit(title, (origin_x, origin_y - 24))

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            rect = pygame.Rect(origin_x + x * GRID_SIZE, origin_y + y * GRID_SIZE, GRID_SIZE - 2, GRID_SIZE - 2)
            draw_ch = ch
            if ch == '@':
                draw_ch = '.'
            pygame.draw.rect(screen, map_tile_color(draw_ch), rect)
            border = (70, 82, 92) if draw_ch == '#' else (125, 135, 140)
            pygame.draw.rect(screen, border, rect, 2 if draw_ch == '#' else 1)

            if ch in OBJECTS or ch == 'B':
                label = font.render(ch, True, (15, 15, 18))
                screen.blit(label, label.get_rect(center=rect.center))

            if ch == '@':
                cx, cy = rect.center
                pygame.draw.circle(screen, (235, 60, 60), (cx, cy), GRID_SIZE // 4)
                angle = {
                    'east': 0,
                    'south': math.pi / 2,
                    'west': math.pi,
                    'north': -math.pi / 2,
                }[world.agent.heading_name]
                tip = (cx + int(math.cos(angle) * 20), cy + int(math.sin(angle) * 20))
                pygame.draw.line(screen, (255, 240, 150), (cx, cy), tip, 4)

    legend_y = origin_y + len(rows) * GRID_SIZE + 18
    legend = 'm mug | p plate | t towel | o bowl | s spoon | l lamp | B bin | @ robot'
    screen.blit(small.render(legend, True, (55, 62, 70)), (origin_x, legend_y))


def draw_slam_minimap(screen, world: World, small):
    origin_x = SLAM_X
    origin_y = SLAM_Y
    rows = world.slam_display_rows()

    title = small.render('SLAM map sent to LLM', True, (35, 42, 50))
    screen.blit(title, (origin_x, origin_y - 22))

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            rect = pygame.Rect(origin_x + x * SLAM_CELL, origin_y + y * SLAM_CELL, SLAM_CELL - 1, SLAM_CELL - 1)
            draw_ch = '.' if ch == '@' else ch
            pygame.draw.rect(screen, slam_tile_color(draw_ch), rect)

            if draw_ch == '#':
                pygame.draw.rect(screen, (65, 76, 86), rect, 1)
            elif draw_ch == '?':
                pygame.draw.rect(screen, (90, 100, 108), rect, 1)
            else:
                pygame.draw.rect(screen, (132, 142, 146), rect, 1)

            if (x, y) == world.spawn_cell and ch != '@':
                pygame.draw.circle(screen, (60, 185, 110), rect.center, max(2, SLAM_CELL // 4), 1)

            if ch in OBJECTS:
                pygame.draw.circle(screen, OBJECTS[ch][1], rect.center, max(2, SLAM_CELL // 3))
            elif ch == 'B':
                pygame.draw.rect(screen, (74, 145, 235), rect.inflate(-3, -3))
            elif ch == '@':
                pygame.draw.circle(screen, (235, 60, 60), rect.center, max(2, SLAM_CELL // 3))

    width = len(rows[0]) * SLAM_CELL if rows else 0
    height = len(rows) * SLAM_CELL
    pygame.draw.rect(screen, (45, 55, 65), (origin_x, origin_y, width, height), 1)
    note = small.render('? unseen | white wall', True, (55, 62, 70))
    screen.blit(note, (origin_x, origin_y + height + 8))


def draw_panel(screen, world: World, state: Dict[str, Any], font, small, mono):
    panel_x = SCREEN_W - PANEL_W
    pygame.draw.rect(screen, (31, 38, 47), (panel_x, 0, PANEL_W, SCREEN_H))
    pygame.draw.line(screen, (132, 145, 158), (panel_x, 0), (panel_x, SCREEN_H), 1)

    x = panel_x + 16
    y = 16
    width = PANEL_W - 32
    controls_top = SCREEN_H - 122

    def line(text: str, color=(235, 238, 242), use_font=None):
        nonlocal y
        surface = (use_font or small).render(text, True, color)
        screen.blit(surface, (x, y))
        y += surface.get_height() + 5

    def block(title: str, text: str, max_lines: int, color=(218, 224, 232)):
        nonlocal y
        if y >= controls_top - 24:
            return
        line(title, (255, 236, 170), font)
        line_height = mono.get_height() + 5
        available_lines = max(0, (controls_top - y - 8) // line_height)
        visible_lines = min(max_lines, available_lines)
        for wrapped in wrap_text(text or '-', mono, width, visible_lines):
            line(wrapped, color, mono)
        y += 8

    line('LLM SLAM Controller', (255, 255, 255), font)
    line(f'Model: {state.get("model_used", OPENROUTER_MODEL)}', (190, 205, 222))
    line(f'Latency: {state["latency"]}s', (190, 225, 195))
    line(f'Queue: {len(state["queue"])} actions', (190, 205, 222))
    robot_room_cell = room_cell((world.agent.x, world.agent.y))
    line(f'Robot: ({robot_room_cell[0]}, {robot_room_cell[1]}) heading {world.agent.heading_name}', (230, 230, 235))
    carrying = 'none' if world.agent.carried is None else OBJECTS[world.agent.carried][0]
    line(f'Carrying: {carrying}', (230, 230, 235))
    delivered = ', '.join(OBJECTS[ch][0] for ch in world.delivered) or 'none'
    line(f'Delivered: {delivered}', (230, 230, 235))
    y += 8

    block('Command', state['input'] if state['input_active'] else state['last_command'], 2)
    block('Status', state['status'], 2, (245, 230, 160))
    block('Sent To LLM', state['prompt_preview'], 5)
    block('Raw LLM Response', state['raw'], 4, (205, 240, 210))
    block('Parsed Actions', str(state['parsed_actions']), 2, (210, 225, 255))
    block('Rationale', state['rationale'], 2)

    controls = [
        'Enter: send typed command once',
        'R: reset world',
        'Backspace: edit command',
        'Esc: quit',
        'LLM sees SLAM map only',
    ]
    y = controls_top
    for item in controls:
        line(item, (180, 188, 200))


def main():
    pygame.init()
    pygame.display.set_caption('LLM SLAM Movement Harness')
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    small = pygame.font.SysFont(None, 19)
    mono = pygame.font.SysFont('consolas', 16)

    world = World()
    executor = ThreadPoolExecutor(max_workers=1)
    future = None
    action_delay = 0

    state: Dict[str, Any] = {
        'input': '',
        'input_active': True,
        'last_command': 'move east two cells',
        'status': world.message,
        'prompt_preview': '',
        'raw': '',
        'parsed_actions': [],
        'rationale': '',
        'latency': 0,
        'model_used': OPENROUTER_MODEL,
        'repair_attempts': 0,
        'llm_turns': 0,
        'task_active': False,
        'executed_actions_for_task': 0,
        'queue': [],
    }

    def request_llm_turn(prompt: str, event_name: str, status: str):
        nonlocal future
        if state['llm_turns'] >= MAX_LLM_TURNS_PER_TASK:
            state['status'] = f'Stopped after {MAX_LLM_TURNS_PER_TASK} LLM turns; task incomplete.'
            state['task_active'] = False
            append_log({
                'event': 'task_stopped',
                'command': state['last_command'],
                'reason': 'max_llm_turns',
                'turns': state['llm_turns'],
            })
            return

        state['llm_turns'] += 1
        state['prompt_preview'] = prompt[:1400] + ('...' if len(prompt) > 1400 else '')
        state['status'] = status
        append_log({
            'event': event_name,
            'command': state['last_command'],
            'turn': state['llm_turns'],
            'prompt': prompt,
        })
        future = executor.submit(ask_llm, prompt, state['last_command'])

    def send_command(command: str):
        nonlocal future
        if future is not None and not future.done():
            state['status'] = 'Still waiting for the previous API response.'
            return
        if not command.strip():
            state['status'] = 'Type a command first.'
            return

        prompt = build_prompt(world, command.strip())
        state['last_command'] = command.strip()
        state['prompt_preview'] = prompt[:1400] + ('...' if len(prompt) > 1400 else '')
        state['raw'] = '(waiting for LLM response)'
        state['parsed_actions'] = []
        state['rationale'] = ''
        state['latency'] = 0
        state['repair_attempts'] = 0
        state['llm_turns'] = 0
        state['task_active'] = True
        state['executed_actions_for_task'] = 0
        state['queue'].clear()
        request_llm_turn(prompt, 'request', 'Sent task to the LLM. Waiting...')

    def poll_future():
        nonlocal future
        if future is None or not future.done():
            return

        result = future.result()
        future = None
        state['latency'] = result['latency']
        state['model_used'] = result.get('model', OPENROUTER_MODEL)

        validated_actions, preflight_error = preflight_actions(world, result['actions'])
        state['raw'] = result['raw']
        state['rationale'] = result['rationale']

        if result.get('used_fallback'):
            state['parsed_actions'] = validated_actions
            state['queue'].extend(validated_actions)
            state['status'] = f"API failed; offline fallback returned {len(validated_actions)} valid action(s). Executing."
        elif preflight_error:
            state['parsed_actions'] = []
            state['repair_attempts'] += 1
            repair_prompt = build_repair_prompt(
                world,
                state['last_command'],
                result['raw'],
                result['actions'],
                preflight_error,
            )
            state['raw'] = result['raw'] + '\n\n(rejected; asking LLM for detour repair)'
            append_log({
                'event': 'repair_request',
                'command': state['last_command'],
                'preflight_error': preflight_error,
                'previous_actions': result['actions'],
                'prompt': repair_prompt,
            })
            request_llm_turn(
                repair_prompt,
                'repair_turn',
                f"LLM route hit a wall. Asking for detour repair {state['repair_attempts']}...",
            )
        else:
            state['parsed_actions'] = validated_actions
            state['queue'].extend(validated_actions)
            state['repair_attempts'] = 0
            state['status'] = f"LLM returned {len(validated_actions)} valid action(s). Executing."
        append_log({
            'event': 'response',
            'command': state['last_command'],
            'raw': result['raw'],
            'actions': result['actions'],
            'validated_actions': validated_actions,
            'preflight_error': preflight_error,
            'rationale': result['rationale'],
            'latency': result['latency'],
            'model': result.get('model', ''),
            'used_fallback': result.get('used_fallback', False),
            'repair_attempts': state['repair_attempts'],
            'error': result.get('error', ''),
        })

    def reset_world():
        nonlocal future, action_delay
        world.reset()
        if future is not None and not future.done():
            future.cancel()
        future = None
        action_delay = 0
        state.update({
            'input': '',
            'last_command': 'move east two cells',
            'status': world.message,
            'prompt_preview': '',
            'raw': '',
            'parsed_actions': [],
            'rationale': '',
            'latency': 0,
            'repair_attempts': 0,
            'llm_turns': 0,
            'task_active': False,
            'executed_actions_for_task': 0,
        })
        state['queue'].clear()

    def finish_task(reason: str):
        state['task_active'] = False
        state['queue'].clear()
        state['status'] = f'SUCCESS: {reason}'
        append_log({
            'event': 'task_complete',
            'command': state['last_command'],
            'reason': reason,
            'room_cell': room_cell((world.agent.x, world.agent.y)),
            'carried': world.agent.carried,
            'delivered': world.delivered,
            'turns': state['llm_turns'],
        })

    def continue_task_if_needed():
        if not state['task_active']:
            return
        if state['queue']:
            return
        if future is not None and not future.done():
            return

        done, reason = task_complete(world, state['last_command'])
        if done:
            finish_task(reason)
            return

        if reason.startswith('no explicit completion') and state['executed_actions_for_task'] > 0:
            finish_task('executed requested movement batch')
            return

        continue_prompt = build_continue_prompt(world, state['last_command'], reason)
        request_llm_turn(
            continue_prompt,
            'continue_request',
            f'Task not complete: {reason}. Asking LLM for next step...',
        )

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RETURN:
                    command = state['input'].strip()
                    state['input'] = ''
                    send_command(command)
                elif event.key == pygame.K_BACKSPACE:
                    state['input'] = state['input'][:-1]
                elif event.key == pygame.K_r and not state['input']:
                    reset_world()
                elif event.unicode and event.unicode.isprintable():
                    state['input'] += event.unicode

        poll_future()

        if state['queue']:
            if action_delay > 0:
                action_delay -= 1
            else:
                action = state['queue'].pop(0)
                world.step(action)
                world.update_slam_from_fov()
                if state['task_active']:
                    state['executed_actions_for_task'] += 1
                state['status'] = world.message
                if world.last_blocked:
                    state['queue'].clear()
                    state['status'] = f'{world.message} Cleared remaining queued moves after collision.'
                append_log({
                    'event': 'sim_action',
                    'action': action,
                    'message': world.message,
                    'room_cell': room_cell((world.agent.x, world.agent.y)),
                    'heading': world.agent.heading_name,
                })
                if state['task_active']:
                    done, reason = task_complete(world, state['last_command'])
                    if done:
                        finish_task(reason)
                action_delay = ACTION_DELAY_FRAMES

        continue_task_if_needed()

        screen.fill((205, 213, 219))
        draw_first_person_view(screen, world, font, small)
        draw_grid(screen, world, font, small)
        draw_slam_minimap(screen, world, small)
        draw_panel(screen, world, state, font, small, mono)
        pygame.display.flip()
        clock.tick(FPS)

    executor.shutdown(wait=False, cancel_futures=True)
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
