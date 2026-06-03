import math
import sys
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set

import pygame


SCREEN_W, SCREEN_H = 960, 540
FPS = 60

TILE_SIZE = 64
FOV = math.radians(60)
NUM_RAYS = 240
MAX_DEPTH = 20.0

MOVE_SPEED = 0.08
TURN_SPEED = 0.05

WORLD_W = 15
WORLD_H = 11
NUM_OBJECTS = 5
NUM_BARRIERS = 3
MAX_GENERATION_ATTEMPTS = 200


OBJECTS = {
    "m": ("mug",   (255, 100,   0)),
    "p": ("plate", ( 23, 255,  31)),
    "t": ("towel", (  0, 140, 255)),
    "o": ("bowl",  (255,  23, 232)),
    "s": ("spoon", (120,  80, 200)),
    "l": ("lamp",  (255, 210,   0)),
}

# Early raycast prototype; main.py has the full agent harness.

def normalize_angle(a: float) -> float:
    return a % (2 * math.pi)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def apply_fog(color: Tuple[int, int, int], depth: float) -> Tuple[int, int, int]:
    fog = clamp(1.0 - (depth / MAX_DEPTH), 0.18, 1.0)
    return tuple(int(c * fog) for c in color)

def is_object_tile(ch: str) -> bool:
    return ch in OBJECTS

def odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def reachable_from_spawn(grid: List[List[str]], spawn: Tuple[int, int]) -> Set[Tuple[int, int]]:
    h = len(grid)
    w = len(grid[0])
    sx, sy = spawn

    if grid[sy][sx] == "#":
        return set()

    q = [(sx, sy)]
    seen = {(sx, sy)}

    while q:
        x, y = q.pop(0)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen:
                if grid[ny][nx] != "#":
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
            if grid[y][x] != ".":
                continue

            n = grid[y - 1][x] == "#"
            s = grid[y + 1][x] == "#"
            e = grid[y][x + 1] == "#"
            wv = grid[y][x - 1] == "#"


            if (n and e) or (n and wv) or (s and e) or (s and wv):
                spots.append((x, y))

    return spots

def add_open_corner_barrier(grid: List[List[str]], x: int, y: int, orientation: str, horiz_len: int, vert_len: int):
    """
    Small L-shaped obstacle that stays open and does not attempt to seal areas.
    """
    h = len(grid)
    w = len(grid[0])

    def set_wall(px: int, py: int):
        if 0 < px < w - 1 and 0 < py < h - 1:
            grid[py][px] = "#"

    gap_index = 1

    if orientation == "right_down":
        for i in range(horiz_len):
            if i != gap_index:
                set_wall(x + i, y)
        for j in range(vert_len):
            if j != gap_index:
                set_wall(x + horiz_len - 1, y + j)

    elif orientation == "down_right":
        for j in range(vert_len):
            if j != gap_index:
                set_wall(x, y + j)
        for i in range(horiz_len):
            if i != gap_index:
                set_wall(x + i, y + vert_len - 1)

    elif orientation == "left_down":
        for i in range(horiz_len):
            if i != gap_index:
                set_wall(x - i, y)
        for j in range(vert_len):
            if j != gap_index:
                set_wall(x - horiz_len + 1, y + j)

    elif orientation == "down_left":
        for j in range(vert_len):
            if j != gap_index:
                set_wall(x, y + j)
        for i in range(horiz_len):
            if i != gap_index:
                set_wall(x - i, y + vert_len - 1)

def generate_simple_world(width: int, height: int, num_objects: int) -> List[str]:
    """
    Generates a small room-like layout:
    - outer walls
    - a few small open corner barriers
    - objects placed only on reachable corner tiles
    """
    width = odd(width)
    height = odd(height)

    for _attempt in range(MAX_GENERATION_ATTEMPTS):
        grid = [["."] * width for _ in range(height)]


        for x in range(width):
            grid[0][x] = "#"
            grid[height - 1][x] = "#"
        for y in range(height):
            grid[y][0] = "#"
            grid[y][width - 1] = "#"


        spawn = (1, 1)
        for yy in range(1, min(4, height - 1)):
            for xx in range(1, min(4, width - 1)):
                grid[yy][xx] = "."

        # A few wall fragments give the camera something to navigate around.
        for _ in range(NUM_BARRIERS):
            x = random.randint(4, width - 5)
            y = random.randint(2, height - 4)

            horiz_len = random.randint(2, 4)
            vert_len = random.randint(2, 4)
            orientation = random.choice(["right_down", "down_right", "left_down", "down_left"])

            add_open_corner_barrier(grid, x, y, orientation, horiz_len, vert_len)

        grid[spawn[1]][spawn[0]] = "A"

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
                    if (x, y) in reachable and grid[y][x] == "." and (x, y) != spawn:
                        n = grid[y - 1][x] == "#"
                        s = grid[y + 1][x] == "#"
                        e = grid[y][x + 1] == "#"
                        wv = grid[y][x - 1] == "#"
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

    raise RuntimeError("Failed to generate a valid reachable map after many attempts.")


@dataclass
class Agent:
    x: float
    y: float
    angle: float
    carried_item: Optional[str] = None

class World:
    def __init__(self, map_rows: List[str]):
        if not map_rows:
            raise ValueError("Map is empty")

        width = len(map_rows[0])
        for row in map_rows:
            if len(row) != width:
                raise ValueError("All map rows must have the same length")

        self.map = [list(row) for row in map_rows]
        self.h = len(self.map)
        self.w = len(self.map[0])
        self.agent, self.spawn = self._find_agent_and_spawn()
        self.done = False
        self.delivered = 0
        self.total_items = sum(1 for row in self.map for ch in row if is_object_tile(ch))
        self.message = "Collect household items and return them to the bin at spawn."

    def _find_agent_and_spawn(self) -> Tuple[Agent, Tuple[int, int]]:
        for y, row in enumerate(self.map):
            for x, ch in enumerate(row):
                if ch == "A":
                    self.map[y][x] = "."
                    return Agent(x + 0.5, y + 0.5, 0.0), (x, y)
        raise ValueError("No agent start 'A' found in map")

    def tile_at(self, mx: int, my: int) -> str:
        if mx < 0 or my < 0 or mx >= self.w or my >= self.h:
            return "#"
        return self.map[my][mx]

    def set_tile(self, mx: int, my: int, ch: str):
        if 0 <= mx < self.w and 0 <= my < self.h:
            self.map[my][mx] = ch

    def is_blocking(self, mx: int, my: int) -> bool:
        return self.tile_at(mx, my) == "#"

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

        if action == "turn_left":
            self.agent.angle = normalize_angle(self.agent.angle - TURN_SPEED)

        elif action == "turn_right":
            self.agent.angle = normalize_angle(self.agent.angle + TURN_SPEED)

        elif action == "move_forward":
            dx = math.cos(self.agent.angle) * MOVE_SPEED
            dy = math.sin(self.agent.angle) * MOVE_SPEED
            self.move(dx, dy)

        elif action == "move_back":
            dx = -math.cos(self.agent.angle) * MOVE_SPEED
            dy = -math.sin(self.agent.angle) * MOVE_SPEED
            self.move(dx, dy)

        elif action == "interact":
            self.try_pickup_or_drop()

        elif action == "wait":
            pass

        self.check_done()

    def try_pickup_or_drop(self):
        mx, my = int(self.agent.x), int(self.agent.y)
        tile = self.tile_at(mx, my)

        if self.agent.carried_item is not None:
            if self.at_spawn():
                item_name = OBJECTS[self.agent.carried_item][0]
                self.delivered += 1
                self.message = f"Dropped {item_name} in the bin. ({self.delivered}/{self.total_items})"
                self.agent.carried_item = None
                self.check_done()
            else:
                self.message = "Carry it back to the bin at spawn."
            return

        if is_object_tile(tile):
            self.agent.carried_item = tile
            item_name = OBJECTS[tile][0]
            self.set_tile(mx, my, ".")
            self.message = f"Picked up {item_name}."
        else:
            self.message = "Nothing to pick up here."

    def check_done(self):
        if self.delivered >= self.total_items and self.total_items > 0:
            self.done = True
            self.message = "All items delivered. Success!"

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
            "position": [round(self.agent.x, 2), round(self.agent.y, 2)],
            "grid_position": [ax, ay],
            "angle_rad": round(self.agent.angle, 3),
            "inventory": [carried_name] if carried_name else [],
            "local_view_5x5": self.get_local_view(radius=2),
            "goal": "Collect household objects and deliver them to the bin at spawn.",
            "bin_location": list(self.spawn),
            "delivered": self.delivered,
            "remaining": remaining + (1 if self.agent.carried_item is not None else 0),
            "message": self.message,
            "done": self.done,
        }


def cast_ray(world: World, ray_angle: float) -> Tuple[float, str, bool]:
    ox, oy = world.agent.x, world.agent.y
    sin_a = math.sin(ray_angle)
    cos_a = math.cos(ray_angle)

    depth = 0.0
    step = 0.02
    hit_tile = "#"
    hit_vertical = False

    while depth < MAX_DEPTH:
        depth += step
        x = ox + cos_a * depth
        y = oy + sin_a * depth
        mx, my = int(x), int(y)

        t = world.tile_at(mx, my)

        # Objects are drawn like short colored walls in this simple renderer.
        if t == "#" or is_object_tile(t):
            hit_tile = t
            frac_x = x - mx
            frac_y = y - my
            hit_vertical = frac_x < frac_y if abs(cos_a) > abs(sin_a) else frac_y < frac_x
            return depth, hit_tile, hit_vertical

    return MAX_DEPTH, hit_tile, hit_vertical

def draw_gradient_background(screen):
    for y in range(SCREEN_H // 2):
        t = y / (SCREEN_H // 2)
        c = (
            int(18 + 12 * t),
            int(18 + 12 * t),
            int(35 + 18 * t),
        )
        pygame.draw.line(screen, c, (0, y), (SCREEN_W, y))

    for y in range(SCREEN_H // 2, SCREEN_H):
        t = (y - SCREEN_H // 2) / (SCREEN_H // 2)
        c = (
            int(35 + 16 * t),
            int(28 + 10 * t),
            int(22 + 8 * t),
        )
        pygame.draw.line(screen, c, (0, y), (SCREEN_W, y))

def render_3d(screen, world: World):
    draw_gradient_background(screen)

    ray_angle = world.agent.angle - FOV / 2
    col_width = SCREEN_W / NUM_RAYS

    for i in range(NUM_RAYS):
        depth, tile, hit_vertical = cast_ray(world, ray_angle)

        corrected_depth = depth * math.cos(world.agent.angle - ray_angle)
        wall_h = int(min(SCREEN_H, (TILE_SIZE * 5) / max(corrected_depth, 0.0001)))
        y0 = SCREEN_H // 2 - wall_h // 2

        if tile == "#":
            base_color = (235, 235, 235)
        elif is_object_tile(tile):
            base_color = OBJECTS[tile][1]
        else:
            base_color = (200, 200, 200)

        shade_color = apply_fog(base_color, corrected_depth)
        if hit_vertical:
            shade_color = tuple(int(c * 0.82) for c in shade_color)

        x = int(i * col_width)
        w = int(math.ceil(col_width))

        pygame.draw.rect(screen, shade_color, (x, y0, w, wall_h))
        edge = tuple(max(0, c - 40) for c in shade_color)
        pygame.draw.rect(screen, edge, (x, y0, 1, wall_h))

        ray_angle += FOV / NUM_RAYS

def draw_minimap(screen, world: World, x=10, y=10, scale=10):
    font = pygame.font.SysFont(None, 18)

    for my, row in enumerate(world.map):
        for mx, ch in enumerate(row):
            if ch == "#":
                c = (90, 90, 90)
            elif is_object_tile(ch):
                c = OBJECTS[ch][1]
            else:
                c = (30, 30, 30)

            pygame.draw.rect(
                screen,
                c,
                (x + mx * scale, y + my * scale, scale - 1, scale - 1),
            )

    bx = x + world.spawn[0] * scale
    by = y + world.spawn[1] * scale
    pygame.draw.rect(screen, (60, 120, 220), (bx, by, scale - 1, scale - 1))

    ax = x + int(world.agent.x * scale)
    ay = y + int(world.agent.y * scale)
    pygame.draw.circle(screen, (220, 60, 60), (ax, ay), max(2, scale // 3))

    fx = ax + int(math.cos(world.agent.angle) * scale)
    fy = ay + int(math.sin(world.agent.angle) * scale)
    pygame.draw.line(screen, (255, 80, 80), (ax, ay), (fx, fy), 2)

    carried = world.agent.carried_item
    label_text = "carrying: none" if carried is None else f"carrying: {OBJECTS[carried][0]}"
    label = font.render(label_text, True, (255, 255, 255))
    screen.blit(label, (x, y + len(world.map) * scale + 4))


def main():
    pygame.init()
    pygame.display.set_caption("LLM 2D World - Simple Corner Objects")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    small = pygame.font.SysFont(None, 18)

    map_rows = generate_simple_world(WORLD_W, WORLD_H, NUM_OBJECTS)
    world = World(map_rows)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            sys.exit()

        if keys[pygame.K_LEFT]:
            world.step("turn_left")
        if keys[pygame.K_RIGHT]:
            world.step("turn_right")
        if keys[pygame.K_UP]:
            world.step("move_forward")
        if keys[pygame.K_DOWN]:
            world.step("move_back")

        if keys[pygame.K_SPACE]:
            world.step("interact")

        render_3d(screen, world)
        draw_minimap(screen, world)

        obs = world.get_observation()
        status_line = (
            f"Pos: {obs['grid_position']} | "
            f"Carry: {obs['inventory']} | "
            f"Delivered: {obs['delivered']}/{world.total_items} | "
            f"{world.message}"
        )
        controls_line = "Arrows move/turn | SPACE pick up / drop | ESC quit"

        text1 = font.render(status_line, True, (255, 255, 255))
        text2 = small.render(controls_line, True, (255, 255, 255))
        screen.blit(text1, (10, SCREEN_H - 48))
        screen.blit(text2, (10, SCREEN_H - 24))

        if world.done:
            done_txt = font.render("SUCCESS", True, (255, 255, 0))
            screen.blit(done_txt, (SCREEN_W - 110, 10))

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
