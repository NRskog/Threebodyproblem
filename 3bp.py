
import numpy as np
import matplotlib.pyplot as plt

#G = 6.67* 10**(-11)
G = 6.67430e-11

class Celestial_body:


    def __init__(self, mass, x_init, y_init, dx, dy, radius):
        self.mass = mass
        self.pos = np.array([x_init, y_init], dtype=float)
        self.velocity = np.array([dx, dy], dtype=float)
        self.radius = radius
        self.orbit = np.array([self.pos])


        self.backup_pos = self.pos.copy()
        self.backup_vel = self.velocity.copy()
        self.orbit_backup = self.orbit.copy()

   # def differential_equations(self, obj1, obj2): ##
    #    r1_dot_dot = G * obj1.mass * (obj2.radius - obj1.radius) / (abs(obj2.radius - obj1.radius)**3)
     #   r2_dot_dot = G * obj1.mass * (obj1.radius - obj2.radius) / (abs(obj1.radius - obj2.radius)**3)

      #  return r1_dot_dot, r2_dot_dot
        #If I use Euler

    def update(self, acceleration, dt):
        self.velocity += acceleration * dt
        self.pos += self.velocity * dt




    def calc_acceleration(self, other_object): #comes from r1_dot_dot = G * obj1.mass * (obj2.radius - obj1.radius) / (abs(obj2.radius - obj1.radius)**3)
        r_vector = other_object.pos - self.pos
        r_norm = np.linalg.norm(r_vector) + 1e-3
        acceleration = G * other_object.mass * (r_vector / (r_norm **3))

        return acceleration
    
    def add_new_pos(self):
        self.orbit = np.vstack([self.orbit, self.pos.copy()]) 

    def copy(self):
        return Celestial_body(self.mass, self.pos[0], self.pos[1], self.velocity[0], self.velocity[1], self.radius)



def sum_of_acceleration(planet, planets):

    net_acc = np.array([0.0,0.0])
    for other_planet in planets:
        if planet != other_planet:
            net_acc += planet.calc_acceleration(other_planet)
    
    return net_acc
def exp_Euler(planet, planets, dt):

    new_acc = sum_of_acceleration(planet, planets)
    planet.velocity += new_acc * dt
    planet.pos  += planet.velocity * dt
    planet.add_new_pos()

def rk3(planet, planets, dt):
    # K1
    k1_acc = sum_of_acceleration(planet, planets)
    k1_vel = planet.velocity
    k1_pos = planet.pos

    # K2
    k2_vel = planet.velocity + 0.5 * k1_acc * dt
    k2_pos = planet.pos + 0.5 * k1_vel * dt
    k2_acc = sum_of_acceleration(planet, planets)

    # K3
    k3_vel = planet.velocity + k2_acc * dt
    k3_pos = planet.pos + k2_vel * dt
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
    min_dt = 500    
    max_dt = 100000   

    # Just to be safe
    planets_rk3 = [p.copy() for p in planets]

    # RK4 on planets
    rk4(planet, planets, dt)

    # RK3 on copies just for comparison
    rk3_planet = next(p for p in planets_rk3 if p.mass == planet.mass)
    rk3(rk3_planet, planets_rk3, dt)

    # calculate error from difference in position at same time RELATIVE ERROR
    error = np.linalg.norm(planet.pos - rk3_planet.pos) / np.linalg.norm(planet.pos)


    # Here are the cases for error and tolerance
    if error > tol:
        # big error make dt smaller
        dt *= 0.9
    elif error < tol / 100:
        # small error, wasting rescources make step bigger
        dt *= 1.2
    # this is ok zone, dont change dt
    elif tol / 2 <= error <= tol:
        pass

    # take the best fitting dt
    dt = max(min(dt, max_dt), min_dt)

    # add pos
    planet.add_new_pos()

    return dt, error
"""

    if error > tol: #Error too large, make step smaller
        dt *= 0.5
    
    elif tol/5 <= error <= tol:
        pass

    elif error < (tol / 5): #Error way too small means we are wasting computation, larger step
        dt *= 2
"""




def simulate(planets, method, steps, dt, tol=1e-3): ##Tol around 1e-3 seems good 1e-2 horrendous
 
    energy_log = []  
    
    for step in range(steps):
        if method == "adaptive RK":
            for planet in planets:
                dt, error = rk_adaptive(planet, planets, dt, tol)
                if step % 500 == 0:  # Debug-utskrift
                    print(f"Step {step}, Current dt: {dt:.5f}, Relative Error: {error:.5e}")
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
        #E_k = mv^2 /2
        kin_energy = 0.5 * planet.mass * np.linalg.norm(planet.velocity)**2

        pot_energy = sum( -G * other_planet.mass / np.linalg.norm(planet.pos - other_planet.pos) for 
        j, other_planet in enumerate(planets) if i != j )

        tot_energy += kin_energy + pot_energy
    return tot_energy


def plot(planets, method, energy_data):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Subplot 1: Banor
    for planet in planets:
        traj = planet.orbit
        axs[0].plot(traj[:, 0], traj[:, 1], label=f"Planet {planet.mass:.2e} kg")
    
    axs[0].set_xlabel("x-position (m)")
    axs[0].set_ylabel("y-position (m)")
    axs[0].legend()
    axs[0].set_title(f"Orbit Simulation using {method}")
    axs[0].axis("equal")

    # Subplot 2: Energi
    steps = range(len(energy_data))  # Stegen är index i listan
    axs[1].plot(steps, energy_data, label="Total Energy", color='b')
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Energy (J)")
    axs[1].legend()
    axs[1].set_title("Energy Conservation")

    plt.tight_layout()
    plt.show()


###TEST


def main(method):

    # sun (central body)
    planet1 = Celestial_body(
    mass=1.989e30,
    x_init=0,
    y_init=0,
    dx=0, dy=0,
    radius=6.957e8
    )
    
    # earth almost (planet)
    planet2 = Celestial_body(
        mass=5.972e27, 
        x_init=1.496e11, 
        y_init=0, 
        dx=0,  
        dy=29783, 
        radius=6.371e6
    )

    #third body (problematic?)
    """
    planet3 = Celestial_body(
        mass=3000,  # Satellitens massa i kg
        x_init=planet2.pos[0] + (6.371e6 + 500e3),  
        y_init=planet2.pos[1], 
        dx=planet2.velocity[0],  
        dy=planet2.velocity[1] + 7660, 
        radius=1.0  
    )
    """
    planets = [planet1, planet2]
    
    energy_log = simulate(planets, method, steps=30000, dt=1300)  
    #print(calculate_energy(planets))
    plot(planets, method, energy_log)
    


if __name__ == "__main__":
    main("RK3")
    