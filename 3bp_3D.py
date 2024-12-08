import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation,FuncAnimation,FFMpegWriter
import imageio_ffmpeg as ffmpeg
import matplotlib as mpl

#G = 6.67* 10**(-11)
G = 6.67430e-11

class Celestial_body:

    def __init__(self, mass, x_init, y_init, z_init, dx, dy, dz, radius):
        self.mass = mass
        self.pos = np.array([x_init, y_init, z_init], dtype=float)
        self.velocity = np.array([dx, dy, dz], dtype=float)
        self.radius = radius
        self.orbit = np.array([self.pos])

        self.backup_pos = self.pos.copy()
        self.backup_vel = self.velocity.copy()
        self.orbit_backup = self.orbit.copy()

    def update(self, acceleration, dt):
        self.velocity += acceleration * dt
        self.pos += self.velocity * dt

    def calc_acceleration(self, other_object):
        r_vector = other_object.pos - self.pos
        r_norm = np.linalg.norm(r_vector) + 1e-3
        acceleration = G * other_object.mass * (r_vector / (r_norm ** 3))
        return acceleration
    
    def add_new_pos(self):
        self.orbit = np.vstack([self.orbit, self.pos.copy()]) 

    def copy(self):
        return Celestial_body(self.mass, self.pos[0], self.pos[1], self.pos[2], 
                              self.velocity[0], self.velocity[1], self.velocity[2], self.radius)

def sum_of_acceleration(planet, planets):
    net_acc = np.array([0.0, 0.0, 0.0])
    for other_planet in planets:
        if planet != other_planet:
            net_acc += planet.calc_acceleration(other_planet)
    return net_acc

def exp_Euler(planet, planets, dt):
    new_acc = sum_of_acceleration(planet, planets)
    planet.velocity += new_acc * dt
    planet.pos += planet.velocity * dt
    planet.add_new_pos()

def rk3(planet, planets, dt):
    # K1
    k1_acc = sum_of_acceleration(planet, planets)
    k1_vel = planet.velocity
    k1_pos = planet.pos

    # K2
    k2_vel = planet.velocity + 0.5 * k1_acc * dt
    k2_pos = planet.pos + 0.5 * k1_vel * dt
    planet.pos, planet.velocity = k2_pos, k2_vel
    k2_acc = sum_of_acceleration(planet, planets)

    # K3
    k3_vel = planet.velocity + k2_acc * dt
    k3_pos = planet.pos + k2_vel * dt
    planet.pos, planet.velocity = k3_pos, k3_vel
    k3_acc = sum_of_acceleration(planet, planets)

    # Uppdatera planetens position och hastighet med RK3
    planet.velocity += (1 / 6) * (k1_acc + 4 * k2_acc + k3_acc) * dt
    planet.pos += (1 / 6) * (k1_vel + 4 * k2_vel + k3_vel) * dt
    planet.add_new_pos()


def rk4(planet, planets, dt):
        #RK4
    # K1
    K1_acc = sum_of_acceleration(planet, planets)
    K1_vel = planet.velocity

    # K2
    K2_vel = planet.velocity + 0.5 * K1_acc * dt
    K2_pos = planet.pos + 0.5 * K1_vel * dt
    planet.pos, planet.velocity = K2_pos, K2_vel
    K2_acc = sum_of_acceleration(planet, planets)

    # K3
    K3_vel = planet.velocity + 0.5 * K2_acc * dt
    K3_pos = planet.pos + 0.5 * K2_vel * dt
    planet.pos, planet.velocity = K3_pos, K3_vel
    K3_acc = sum_of_acceleration(planet, planets)

    # K4
    K4_vel = planet.velocity + K3_acc * dt
    K4_pos = planet.pos + K3_vel * dt
    planet.pos, planet.velocity = K4_pos, K4_vel
    K4_acc = sum_of_acceleration(planet, planets)

    # Updates planet pos and velocity abd adds lates position to list of orbital values
    planet.velocity += (1 / 6) * (K1_acc + 2 * K2_acc + 2 * K3_acc + K4_acc) * dt
    planet.pos += (1 / 6) * (K1_vel + 2 * K2_vel + 2 * K3_vel + K4_vel) * dt
    planet.add_new_pos()

def rk_adaptive(planet, planets, dt, tol):
    min_dt = 2000    
    max_dt = 100000    

    # Backup planets for RK3
    planets_rk3 = [p.copy() for p in planets]

    # RK4
    rk4(planet, planets, dt)

    # RK3 on backup planets
    rk3_planet = next(p for p in planets_rk3 if p.mass == planet.mass)
    rk3(rk3_planet, planets_rk3, dt)

    # Calculate relative error
    error = np.linalg.norm(planet.pos - rk3_planet.pos) / (np.linalg.norm(planet.pos) + 1e-8)

    if error > tol:
        dt *= 0.9
    elif error < tol / 100:
        dt *= 1.2
    dt = max(min(dt, max_dt), min_dt)
    planet.add_new_pos()
    return dt, error

def simulate(planets, method, steps, dt, tol=1e-3):
    energy_log = []
    for step in range(steps):
        if method == "adaptive RK":
            for planet in planets:
                dt, error = rk_adaptive(planet, planets, dt, tol)
                if step % 500 == 0:  # Debug-utskrift
                    #print(f"Step {step}, Current dt: {dt:.5f}, Relative Error: {error:.5e}")
                    pass

        else:
            for planet in planets:
                if method == "RK4":
                    rk4(planet, planets, dt)
                elif method == "RK3":
                    rk3(planet, planets, dt)
                elif method == "EE":
                    exp_Euler(planet, planets, dt)

        energy = calculate_energy(planets)
        energy_log.append(energy)
    return energy_log



def calculate_energy(planets):
    tot_energy = 0
    for i, planet in enumerate(planets):
        kin_energy = 0.5 * planet.mass * np.linalg.norm(planet.velocity)**2
        pot_energy = sum(
            -G * other_planet.mass / np.linalg.norm(planet.pos - other_planet.pos)
            for j, other_planet in enumerate(planets) if i != j
        )
        tot_energy += kin_energy + pot_energy
    return tot_energy



def plot(planets, method, energy_data):
    fig = plt.figure(figsize=(12, 12))
    
    # 3D Trajectory Plot
    ax1 = fig.add_subplot(211, projection='3d')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']  # Manually defined colors
    for idx, planet in enumerate(planets):
        traj = planet.orbit
        color = colors[idx % len(colors)]  # Cycle through defined colors
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f"Planet {planet.mass:.2e} kg", color=color)
        
        # Highlight the final position as a larger marker
        final_pos = traj[-1]
        ax1.scatter(final_pos[0], final_pos[1], final_pos[2], color=color, s=100, label=f"Final Position ({planet.mass:.2e} kg)")

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('z (m)')
    ax1.set_title(f"3D Orbit Simulation using {method}")
    ax1.legend()

    # Energy Plot
    ax2 = fig.add_subplot(212)
    steps = range(len(energy_data))  # Steps are indices in energy data
    ax2.plot(steps, energy_data, label="Total Energy", color='b')
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Energy (J)")
    ax2.legend()
    ax2.set_title("Energy Conservation")

    plt.tight_layout()
    plt.show()


def animate(planets,method, steps, frame_skip=50, max_frames = 15000):

    if method == "adaptive RK":
        frame_skip = frame_skip * 3  # skips more frames for adaptive RK
    
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Fcolor for each planet
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    trajectories = []
    planets_dots = []

    #  create traj and dot for each planet
    for i, planet in enumerate(planets):
        line, = ax.plot([], [], [], '-', color=colors[i % len(colors)], label=f"Planet {i + 1}")
        dot, = ax.plot([], [], [], 'o', color=colors[i % len(colors)], markersize=8)
        trajectories.append(line)
        planets_dots.append(dot)
    
    # axis settings
    ax.set_xlim(-1.5e11, 1.5e11)
    ax.set_ylim(-1.5e11, 1.5e11)
    ax.set_zlim(-1.5e11, 1.5e11)
    ax.set_xlabel('x-position (m)')
    ax.set_ylabel('y-position (m)')
    ax.set_zlabel('z-position (m)')
    ax.set_title(f"3D Simulation Animation ({method})")
    ax.legend()

    # Position data
    positions = [planet.orbit for planet in planets]

    def update(frame):
        frame *= frame_skip  # skip some frams
        if frame >= steps:
            frame = steps - 1  # SKip last fram
        for i, planet in enumerate(planets):
            x_current = positions[i][:frame + 1, 0]
            y_current = positions[i][:frame + 1, 1]
            z_current = positions[i][:frame + 1, 2]

            # Uppdate traj
            trajectories[i].set_data(x_current, y_current)
            trajectories[i].set_3d_properties(z_current)

            # update dot
            planets_dots[i].set_data([x_current[-1]], [y_current[-1]])
            planets_dots[i].set_3d_properties([z_current[-1]])
        return trajectories + planets_dots

    anim = FuncAnimation(fig, update, frames=max_frames// frame_skip, interval=50, blit=False)


    # Save as video
    output_file = f'C:\\Users\\ruben\\Documents\\3bp\\animation_{method}.mp4'
    writer = mpl.animation.FFMpegWriter(fps=30, metadata={'title': f'{method} Animation'})
    anim.save(output_file, writer=writer)
    print(f"Animation saved as {output_file}")
    
    plt.show()
   

 

def main(method):
    # Sun (central body)
    planet1 = Celestial_body(
        mass=1.989e30,
        x_init=0, y_init=0, z_init=0,
        dx=0, dy=0, dz=0,
        radius=6.957e8
    )

    # Earth (planet)
    planet2 = Celestial_body(
        mass=5.972e27,
        x_init=1.496e11, y_init=0, z_init=0,
        dx=0, dy=29783, dz=0,
        radius=6.371e6
    )

    # Satellite
    
    planet3 = Celestial_body(
        mass=5.972e27,
        x_init=0, y_init=0, z_init=1.496e11,
        dx=29783, dy=0, dz=0,
        radius=6.371e6
    )
    

    planets = [planet1, planet2, planet3]

    steps=30000
    dt=1300

    energy_log = simulate(planets, method, steps, dt)
    #plot(planets, method, energy_log)
    animate(planets, method, steps)

if __name__ == "__main__":
    main("adaptive RK")
