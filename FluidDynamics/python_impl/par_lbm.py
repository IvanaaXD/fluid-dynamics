import numpy as np
import time
from multiprocessing import Process, RawArray, Barrier, cpu_count

# 1. Konstante D2Q9 modela
WIDTH, HEIGHT = 400, 100
RELAXATION_TIME = 0.6
ITERATIONS = 1000
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])

def worker_task(rank, num_processes, f_shared_ptr, shape, barrier):
    """
    zadatak koji izvrsava svaki pojedinacni proces nad dijeljenom memorijom
    """
    # povezivanje sa djeljenom memorijom (Shared Memory) preko NumPy pogleda (view)
    # np.frombuffer ne kopira podatke, vec kreira NumPy interfejs nad RawArray-om
    f = np.frombuffer(f_shared_ptr).reshape(shape)
    
    # DOMAIN DECOMPOSITION: podjela mreze na horizontalne trake (stripes)
    # svaki proces dobija fiksni opseg redova (y-osa) na kojima radi
    rows_per_process = shape[1] // num_processes
    start_y = rank * rows_per_process
    end_y = (rank + 1) * rows_per_process if rank != num_processes - 1 else shape[1]

    for it in range(ITERATIONS):
        # 1. COLLISION KORAK (Paralelno racunanje sudara)
        # operacije se vrse samo nad dodjeljenim start_y:end_y opsegom redova
        rho = np.sum(f[:, start_y:end_y, :], axis=0)
        # numericka zastita od djeljenja nulom (division by zero)
        rho[rho == 0] = 1.0 
        
        # proracun makroskopskih brzina (vektorizovano pomocu NumPy-a)
        ux = np.sum(f[:, start_y:end_y, :] * CX[:, None, None], axis=0) / rho
        uy = np.sum(f[:, start_y:end_y, :] * CY[:, None, None], axis=0) / rho
        u_sq = ux**2 + uy**2
        
        # relaksacija distribucija ka ekvilibrijumu (BGK operator)
        for i in range(9):
            u_dot_c = CX[i] * ux + CY[i] * uy
            feq = rho * W[i] * (1 + 3*u_dot_c + 4.5*u_dot_c**2 - 1.5*u_sq)
            # direktna izmjena podataka u djeljenoj memoriji
            f[i, start_y:end_y, :] += -(1.0 / RELAXATION_TIME) * (f[i, start_y:end_y, :] - feq)

        # BARRIER SYNCHRONIZATION: prva barijera osigurava da svi procesi zavrse 
        # Collision pre nego sto streaming faza pocne da mijenja vrijednosti susjeda
        barrier.wait()

        # 2. STREAMING KORAK 
        # druga barijera sluzi za uskladjenost sa specifikacijom i sinhronizovano 
        # napredovanje u sljedecu iteraciju simulacije
        barrier.wait()

def run_parallel_simulation():
    # automatsko detektovanje broja dostupnih logickih jezgara (npr. 16 za Ryzen 7 5800H)
    num_processes = cpu_count()
    shape = (9, HEIGHT, WIDTH)
    # RawArray zahtjeva osnovni Python int tip podatka
    size = int(np.prod(shape)) 
    
    # KREIRANJE DJELJENE MEMORIJE: 'd' oznacava 'double' (f64)
    # ovo omogucava da svi procesi vide iste podatke bez skupog slanja poruka (IPC)
    f_shared_ptr = RawArray('d', size)
    f_init = np.frombuffer(f_shared_ptr).reshape(shape)
    
    # inicijalno punjenje sistema (uniformno stanje)
    f_init[:] = 1.0 / 9.0 
    
    # barijera za sinhronizaciju n-procesa (Critical Path)
    barrier = Barrier(num_processes)
    processes = []
    
    print(f"Running Python Shared Memory simulations with {num_processes} processes...")
    start_time = time.time()
    
    # kreiranje i pokretanje paralelnih procesa
    for i in range(num_processes):
        p = Process(target=worker_task, args=(i, num_processes, f_shared_ptr, shape, barrier))
        p.start()
        processes.append(p)
        
    # cekanje da svi procesi zavrse rad (Join)
    for p in processes:
        p.join()
        
    end_time = time.time()
    print(f"Python Shared Memory ({num_processes} processes) done for: {end_time - start_time:.2f} s")

if __name__ == "__main__":
    run_parallel_simulation()