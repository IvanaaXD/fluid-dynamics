use std::env; 
mod sequential;
mod parallel;
mod visualizer;

fn main() {
    // prikupljanje argumenata komandne linije prosleđenih preko 'cargo run' ili skripte
    let args: Vec<String> = env::args().collect();
    
    // parsiranje parametara sa podrazumijevanim vrijednostima (default values)
    // Argument 1: broj niti (Threads)
    let threads: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    // Argument 2: sirina mreze (Width) - utice na rezoluciju i Weak Scaling
    let width: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(400);
    // Argument 3: visina mreze (Height)
    let height: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
    
    // LOGIKA ZA IZBOR MODULA:
    // ako je broj niti 1, koristimo namjenski sekvencijalni modul bez paralelnog overhead-a
    // ovo je kljucno za precizno izracunavanje Amdalovog Speedup-a
    if threads == 1 {
        println!("Pokretanje sekvencijalne verzije (Baseline)...");
        // sequential.rs koristi f.clone() i jednostavan streaming model
        sequential::run_sequential(width, height, 1000);
    } else {
        println!("Pokretanje paralelne verzije sa {} niti (HPC Mode)...", threads);
        // parallel.rs koristi Rayon, Loop Fusion i Pointer Swap za maksimalne performanse
        parallel::run_parallel(width, height, 1000, threads);
    }

    // Vizuelizacija rezultata:
    // podaci se citaju iz generisanog .dat fajla i pretvaraju u PNG toplotnu mapu
    println!("Generisanje slike strujanja fluida...");
    visualizer::create_visualization(
        "../data/rust_seq_state.dat", 
        "../reports/fluid_flow.png", 
        width, 
        height
    );
}