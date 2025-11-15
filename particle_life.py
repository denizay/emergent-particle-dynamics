import random
import time

import imageio.v2 as imageio
import numpy as np
import pygame
from numba import jit, prange

WIDTH = 1400
FPS = 60
RECORD = False
OUTPUT_FILE = "simulation.mp4"
FRAME_LIMIT = 10

POWER = 30
DAMPING = 0.999

NUM_PARTICLES = 32000

HEIGHT = int(WIDTH * 0.5625)

colors = ["green", "red", "blue", "yellow", "pink"]
COLOR_TO_IDX = {color: idx for idx, color in enumerate(colors)}
DIRECTION_MATRIX = np.round(
    np.random.uniform(-1, 1, size=(len(colors), len(colors))), 1
)


@jit(nopython=True, parallel=True, fastmath=True)
def calculate_forces(positions, color_indices, acc_x, acc_y, power, direction_matrix):
    """
    Optimized force calculation using Numba JIT compilation.
    Uses parallel execution and vectorized operations.
    """
    n = positions.shape[0]

    # Reset accelerations
    acc_x[:] = 0
    acc_y[:] = 0

    for i in prange(n):
        for j in range(n):
            if i == j:
                continue

            # Calculate distance
            # We can cache the distances
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dist_sq = dx * dx + dy * dy

            # Early exit if too far (50^2 = 2500)
            if dist_sq == 0 or dist_sq > 2500:
                continue

            dist = np.sqrt(dist_sq)

            # TO DO: They should be able to pull/push each other from the other side of the map

            if dist < 10:
                # Repel if too close
                force = -0.1 / max(1, dist)
            else:
                # Get direction based on color interaction
                direction = direction_matrix[color_indices[i], color_indices[j]]
                force = power * direction / dist**2

            acc_x[i] += dx * force
            acc_y[i] += dy * force


def create_particles(num_particles):
    particles = {}
    for idx in range(num_particles):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        color = random.choice(colors)
        size = 2

        particles[idx] = {
            "color": color,
            "position": pygame.Vector2(x, y),
            "vel_x": 0,
            "vel_y": 0,
            "acc_x": 0,
            "acc_y": 0,
            "size": size,
        }

    return particles


def particles_to_arrays(particles):
    """Convert particle dict to numpy arrays for Numba processing."""
    n = len(particles)
    positions = np.zeros((n, 2), dtype=np.float64)
    velocities = np.zeros((n, 2), dtype=np.float64)
    accelerations = np.zeros((n, 2), dtype=np.float64)
    color_indices = np.zeros(n, dtype=np.int32)

    for i, (pid, p) in enumerate(particles.items()):
        positions[i] = [p["position"].x, p["position"].y]
        velocities[i] = [p["vel_x"], p["vel_y"]]
        color_indices[i] = COLOR_TO_IDX[p["color"]]

    return positions, velocities, accelerations, color_indices


def arrays_to_particles(particles, positions, velocities):
    """Update particle dict from numpy arrays."""
    for i, (pid, p) in enumerate(particles.items()):
        p["position"].x = positions[i, 0]
        p["position"].y = positions[i, 1]
        p["vel_x"] = velocities[i, 0]
        p["vel_y"] = velocities[i, 1]


@jit(nopython=True, fastmath=True)
def update_positions(positions, velocities, accelerations, dt, damping, width, height):
    """Optimized position update using Numba."""
    n = positions.shape[0]

    for i in range(n):
        # velocities[i, 0] += accelerations[i, 0] * dt
        # velocities[i, 1] += accelerations[i, 1] * dt

        # velocities[i, 0] *= damping
        # velocities[i, 1] *= damping

        # positions[i, 0] += velocities[i, 0] * dt
        # positions[i, 1] += velocities[i, 1] * dt

        # Just move the particles | better visuals
        positions[i, 0] += accelerations[i, 0] * dt
        positions[i, 1] += accelerations[i, 1] * dt

        # Wrap around boundaries
        if positions[i, 0] < 0:
            positions[i, 0] += width
        elif positions[i, 0] > width:
            positions[i, 0] -= width

        if positions[i, 1] < 0:
            positions[i, 1] += height
        elif positions[i, 1] > height:
            positions[i, 1] -= height


time_forces, time_update, time_sync = 0, 0, 0


def update_particle_positions_optimized(
    particles, positions, velocities, accelerations, color_indices, dt
):
    """Optimized update using Numba-compiled functions."""
    global time_forces, time_update, time_sync
    # Calculate forces
    start_time = time.time()
    calculate_forces(
        positions,
        color_indices,
        accelerations[:, 0],
        accelerations[:, 1],
        POWER,
        DIRECTION_MATRIX,
    )
    end_time = time.time()
    time_forces += end_time - start_time
    start_time = time.time()
    # Update positions
    update_positions(positions, velocities, accelerations, dt, DAMPING, WIDTH, HEIGHT)
    end_time = time.time()
    time_update += end_time - start_time
    start_time = time.time()
    # Sync back to particle dict for rendering
    arrays_to_particles(particles, positions, velocities)
    end_time = time.time()
    time_sync += end_time - start_time


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    if RECORD:
        frames = []

    particles = create_particles(NUM_PARTICLES)

    # Convert to arrays for Numba processing
    positions, velocities, accelerations, color_indices = particles_to_arrays(particles)

    running = True
    dt = 0

    start_time = time.time()
    count = 0
    while running:
        count += 1
        if FRAME_LIMIT and count > FRAME_LIMIT:
            running = False
            end_time = time.time()
            print(
                f"Time taken for {FRAME_LIMIT} frames: {end_time - start_time} seconds"
            )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill("black")

        for particle in particles.values():
            # print(type(particle))
            pygame.draw.circle(
                screen, particle["color"], particle["position"], particle["size"]
            )

        update_particle_positions_optimized(
            particles, positions, velocities, accelerations, color_indices, dt
        )

        pygame.display.flip()

        if RECORD:
            frame_data = pygame.surfarray.array3d(screen)
            frame_data = np.transpose(frame_data, (1, 0, 2))
            frames.append(frame_data)

        dt = clock.tick(FPS) / 1000

    if RECORD:
        imageio.mimsave(OUTPUT_FILE, frames, fps=FPS)

    pygame.quit()
    print(f"Time taken for forces: {time_forces} seconds")
    print(f"Time taken for update: {time_update} seconds")
    print(f"Time taken for sync: {time_sync} seconds")


if __name__ == "__main__":
    # import cProfile
    # import pstats

    # profiler = cProfile.Profile()
    # profiler.enable()

    main()

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative')
    # stats.print_stats(20)
