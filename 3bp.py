
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

def sum_of_acceleration(planet, planets):

    net_acc = np.array([0.0,0.0])
    for other_planet in planets:
        if planet != other_planet:
            net_acc += planet.calc_acceleration(other_planet)
    
    return net_acc


def rk(planet, planets, dt):
        #RK4
    #K1
    #K1_pos = planet.pos
    K1_vel = planet.velocity
    K1_acc = sum_of_acceleration(planet, planets)
    
    
    #K2
    #K2_pos = planet.pos + 0.5 * K1_vel * dt
    K2_vel = planet.velocity + 0.5 * K1_acc * dt
    K2_acc = sum_of_acceleration(planet, planets)

    #K3
    #K3_pos = planet.pos + 0.5 * K2_vel * dt
    K3_vel = planet.velocity * 0.5 * K2_acc * dt
    K3_acc = sum_of_acceleration(planet, planets)   


    #K4
    #K4_pos = planet.pos +  K3_vel * dt
    K4_vel = planet.velocity +  K3_acc * dt
    K4_acc = sum_of_acceleration(planet, planets)  
    
    #These two are y_n+1 = y_n + 1/6(k1 + 2k2 +2k3 + k4)
    planet.velocity += (1/6) * (K1_acc + 2*K2_acc + 2*K3_acc + K4_acc) * dt         
    planet.pos += (1/6) * (K1_vel + 2*K2_vel + 2*K3_vel + K4_vel) * dt
    
    planet.add_new_pos()



def simulate(planets, steps, method, dt):

    if method == "RK":
        for _ in range(steps):
            for planet in planets:
                rk(planet, planets, dt)


def plot(planets):
    for planet in planets:
        traj = planet.orbit
        plt.plot(traj[:, 0], traj[:, 1], label=f"Planet {planet.mass:.2e} kg")
    
    plt.xlabel("x-position (m)")
    plt.ylabel("y-position (m)")
    plt.legend()
    plt.autoscale()  # Anpassa gränser automatiskt

    plt.axis("equal")  # Symmetrisk skalning
    plt.show()


###TEST


def main():

    # Solen (central kropp)
    planet1 = Celestial_body(mass=1.989e30, x_init=0, y_init=0, dx=0, dy=0, radius=6.957e8)
    
    # Jorden (planet)
    planet2 = Celestial_body(
        mass=5.972e24, 
        x_init=1.496e11, 
        y_init=0, 
        dx=0,  # Tangentiell hastighet
        dy=29783, 
        radius=6.371e6
    )
    
    planets = [planet1, planet2]
    
    # Simulera med bättre tidssteg och fler iterationer
    simulate(planets, steps=10000, method="RK", dt=3600)  # 1 timmes tidssteg
    
    # Plotta med justerade gränser
    plot(planets)


if __name__ == "__main__":
    main()
    