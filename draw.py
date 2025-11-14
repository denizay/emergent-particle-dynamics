import colorsys
import random

import imageio.v2 as imageio  # v2 API is more stable
import numpy as np
import pygame

WIDTH = 1600
FPS = 60
RECORD = False
OUTPUT_FILE = "simulation.mp4"

POWER = 300000
DAMPING = 0.999
NUM_PARTICLES = 8

HEIGHT = int(WIDTH * 0.5625)


def pastel_planet_color():
    h = random.random()
    s = random.uniform(0.10, 0.30)
    v = random.uniform(0.80, 0.95)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def create_particles(num_particles):
    particles = {}
    for idx in range(num_particles):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        color = pastel_planet_color()
        # color = "white"

        # choose normal distribution for size
        size = np.random.normal(30, 20)
        size = int(size)
        size = max(2, size)
        # size = min(200, size)

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


def calculate_particle_adds(particle_id, particle, particles):
    for other_particle_id, other_particle in particles.items():
        if other_particle_id == particle_id:
            continue

        difference = other_particle["position"] - particle["position"]
        distance = difference.length()
        if distance == 0:
            continue

        difference.normalize_ip()

        threshold = other_particle["size"] + particle["size"]
        force = POWER * other_particle["size"] / max(1, distance**2)

        if distance > threshold:
            particle["acc_x"] += difference.x * force
            particle["acc_y"] += difference.y * force


def update_particle_position(particle, dt):
    particle["vel_x"] += particle["acc_x"] * dt
    particle["vel_y"] += particle["acc_y"] * dt

    particle["vel_x"] *= DAMPING
    particle["vel_y"] *= DAMPING

    particle["position"].x += particle["vel_x"] * dt
    particle["position"].y += particle["vel_y"] * dt

    particle["acc_x"] = 0
    particle["acc_y"] = 0


def update_particle_positions(particles, dt):
    for particle_id, particle in particles.items():
        calculate_particle_adds(particle_id, particle, particles)
    for particle in particles.values():
        update_particle_position(particle, dt)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    if RECORD:
        frames = []

    particles = create_particles(NUM_PARTICLES)

    # ADD the "sun" optionally
    # particles[0] = {
    #     "color": "yellow",
    #     "position": pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2),
    #     "vel_x": 0,
    #     "vel_y": 0,
    #     "acc_x": 0,
    #     "acc_y": 0,
    #     "size": 150,
    # }

    running = True
    dt = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill("black")

        for particle in particles.values():
            pygame.draw.circle(
                screen, particle["color"], particle["position"], particle["size"]
            )

        update_particle_positions(particles, dt)

        pygame.display.flip()

        if RECORD:
            frame_data = pygame.surfarray.array3d(screen)
            frame_data = np.transpose(frame_data, (1, 0, 2))
            frames.append(frame_data)

        dt = clock.tick(FPS) / 1000

    if RECORD:
        imageio.mimsave(OUTPUT_FILE, frames, fps=FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
