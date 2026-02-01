import numpy as np
import time

# 1. Konstante D2Q9 modela (2 dimenzije, 9 diskretnih brzina)
WIDTH, HEIGHT = 200, 50
RELAXATION_TIME = 0.6  # Tau: odredjuje brzinu povratka u ravnotezu i viskoznost fluida
ITERATIONS = 1000

# tezinski faktori za svaki od 9 pravaca (centar, kardinalni pravci, dijagonale)
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# vektori diskretnih brzina (lattice velocities) ci
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])

# indeksi suprotnih smjerova za implementaciju Bounce-back uslova na prepreci
OPPOSITE = [0, 3, 4, 1, 2, 7, 8, 5, 6]

def get_equilibrium(rho, u):
    """
    izracunava lokalnu ravnoteznu distribuciju (Maxwell-Boltzmann equilibrium)
    na osnovu trenutne gustine i brzine u svakom čvoru
    """
    feq = np.zeros((9, HEIGHT, WIDTH))
    for i in range(9):
        # skalarni proizvod u * ci: projektuje makroskopsku brzinu na diskretne pravce
        u_dot_c = CX[i] * u[0] + CY[i] * u[1]
        u_sq = u[0]**2 + u[1]**2
        # BGK aproksimacija ekvilibrijuma (kvadratni oblik po brzini u)
        feq[i] = rho * W[i] * (1 + 3*u_dot_c + 4.5*u_dot_c**2 - 1.5*u_sq)
    return feq

def run_simulation():
    # inicijalizacija polja: rho (gustina) je uniformna, fluid tece lagano udesno
    rho = np.ones((HEIGHT, WIDTH))
    u = np.zeros((2, HEIGHT, WIDTH))
    u[0] = 0.1 # pocetni "wind" u x smeru
    
    # postavljanje pocetnih distribucija cestica u ravnotezno stanje
    f = get_equilibrium(rho, u)
    
    # kreiranje binarne maske za prepreku (cilindar)
    obstacle = np.zeros((HEIGHT, WIDTH), dtype=bool)
    center_x, center_y, radius = WIDTH//4, HEIGHT//2, HEIGHT//10
    y, x = np.ogrid[:HEIGHT, :WIDTH]
    obstacle[(x - center_x)**2 + (y - center_y)**2 <= radius**2] = True

    start_time = time.time()

    for it in range(ITERATIONS):
        # 1. STREAMING STEP (Advekcija)
        # pomjeranje vjerovatnoća distribucije cestica u susjedne cvorove mreze
        # np.roll simulira periodicne uslove granica (periodic boundaries)
        for i in range(9):
            f[i] = np.roll(f[i], shift=(CY[i], CX[i]), axis=(0, 1))
        
        # 2. BOUNDARY CONDITIONS (Bounce-back)
        # cestice koje udare u prepreku se odbijaju u suprotnom smjeru (No-slip condition)
        f_at_obstacle = f[:, obstacle]
        f[:, obstacle] = f_at_obstacle[OPPOSITE]
        
        # 3. IZRACUNAVANJE MAKROSKOPSKIH VELICINA
        # gustina je suma svih distribucija u cvoru, a brzina je tezinska suma pravaca
        rho = np.sum(f, axis=0)
        # CX[:, None, None] dodaje dimenzije kako bi se nizovi poklopili sa f (broadcasting)
        u[0] = np.sum(f * CX[:, None, None], axis=0) / rho
        u[1] = np.sum(f * CY[:, None, None], axis=0) / rho
        
        # forsiranje brzine na nulu unutar objekta (dodatna numericka stabilnost)
        u[:, obstacle] = 0
        
        # 4. COLLISION STEP (BGK operator)
        # relaksacija distribucija ka ravnoteznom stanju (feq)
        feq = get_equilibrium(rho, u)
        # f(t+1) = f(t) - (1/tau) * (f(t) - feq)
        f += -(1.0 / RELAXATION_TIME) * (f - feq)
        
        if it % 100 == 0:
            print(f"Iteration {it} done")

    end_time = time.time()
    
    # cuvanje finalne magnitude brzine (velicina vektora |u|)
    # numpy.save je brza binarna alternativa cuvanju u tekstualne fajlove
    velocity_magnitude = np.sqrt(u[0]**2 + u[1]**2)
    np.save("../data/final_state.npy", velocity_magnitude)
    
    print(f"Sequential time (Python): {end_time - start_time:.2f} s")

if __name__ == "__main__":
    run_simulation()