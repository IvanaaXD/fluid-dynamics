import subprocess
import time
import numpy as np
import csv
import os
import matplotlib.pyplot as plt

# konfiguracija eksperimenta: broj ponavljanja za statisticku stabilnost
SAMPLES = 30
# razlicite konfiguracije niti koje testiramo (1 do 16 za Ryzen 7 5800H)
THREAD_CONFIGS = [1, 2, 4, 8, 16]

def run_cmd(threads, width, height):
    """izvrsava rust binarni fajl sa prosledjenim argumentima i mjeri vrijeme"""
    start = time.perf_counter()
    # pozivamo rust implementaciju preko cargo run komande
    subprocess.run(
        ["cargo", "run", "--release", str(threads), str(width), str(height)], 
        cwd="./rust_impl", capture_output=True, text=True
    )
    return time.perf_counter() - start

def execute_experiment(name, is_weak):
    """
    sprovodi set testova za strong ili weak scaling
    rezultate cuva u csv fajl radi kasnije analize
    """
    filename = f"reports/rust_{name}_scaling.csv"
    os.makedirs('reports', exist_ok=True)
    
    results = []
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # zaglavlje csv fajla sa statistickim parametrima
        writer.writerow(["Threads", "Width", "Height", "Mean_Time_s", "Std_Dev", "Outliers"])

        for t in THREAD_CONFIGS:
            # ako je is_weak=True, sirina mreze raste linearno sa brojem niti (Gustafson)
            # ako je is_weak=False, dimenzije ostaju fiksne na 400x100 (Amdahl)
            w = 200 * t if is_weak else 400
            h = 100
            
            times = []
            print(f"Tetsing {name}: {t} thread ({w}x{h})...", end=" ", flush=True)
            for _ in range(SAMPLES):
                times.append(run_cmd(t, w, h))
            
            # statisticka obrada podataka: srednja vrijednost i standardna devijacija
            mean_v, std_v = np.mean(times), np.std(times)
            # detekcija iskocenih vrijednosti (outliers) izvan opsega od 2 standardne devijacije
            outliers = [round(x, 4) for x in times if abs(x - mean_v) > 2 * std_v]
            
            writer.writerow([t, w, h, round(mean_v, 4), round(std_v, 4), outliers])
            print(f"Middle: {mean_v:.4f}s")
            results.append((t, mean_v))
    return results

def generate_plots(strong_data, weak_data):
    """generise uporedne grafikone na osnovu prikupljenih mjerenja"""
    print("Generating graphics...")
    
    # 1. Grafik za Strong Scaling (Amdahl's Law)
    threads_s, times_s = zip(*strong_data)
    # speedup se racuna kao T(1) / T(n)
    base_time = times_s[0]
    speedup = [base_time / t for t in times_s]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # plava linija predstavlja tvoje izmjereno ubrzanje
    plt.plot(threads_s, speedup, 'o-', label='Measured Speedup', color='blue')
    # isprekidana linija predstavlja savrseno linearno ubrzanje (idealan scenario)
    plt.plot(threads_s, threads_s, '--', label='Ideal (Linearno)', color='gray')
    plt.title('Strong Scaling (Amdahl)')
    plt.xlabel('Number of threads')
    plt.ylabel('Acceleration (Speedup)')
    plt.legend()
    plt.grid(True)

    # 2. Grafik za Weak Scaling (Gustafson's Law)
    # ovdje pratimo kako vrijeme ostaje (ili ne ostaje) konstantno dok raste i posao i broj jezgara
    threads_w, times_w = zip(*weak_data)
    plt.subplot(1, 2, 2)
    plt.plot(threads_w, times_w, 's-', label='Execution time', color='red')
    plt.title('Weak Scaling (Gustafson)')
    plt.xlabel('Number of threads (The workload is increasing)')
    plt.ylabel('Time (s)')
    plt.grid(True)

    plt.tight_layout()
    # cuvanje grafika kao PNG slike za potrebe izvjestaja
    plt.savefig('reports/scaling_plots.png')
    print("Graphics saved in reports/scaling_plots.png")

if __name__ == "__main__":
    # pokretanje oba eksperimenta redom
    strong_results = execute_experiment("strong", is_weak=False)
    weak_results = execute_experiment("weak", is_weak=True)
    generate_plots(strong_results, weak_results)