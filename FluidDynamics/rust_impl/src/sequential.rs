use ndarray::{Array2, Array3};
use std::fs::File;
use std::io::Write;

// D2Q9 model konstante: tezinski faktori (W) i vektori brzina (CX, CY)
const W: [f64; 9] = [4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0];
const CX: [isize; 9] = [0, 1, 0, -1, 0, 1, -1, -1, 1];
const CY: [isize; 9] = [0, 0, 1, 0, -1, 1, 1, -1, -1];
// indeksi suprotnih smjerova za bounce-back (odbijanje od prepreke)
const OPPOSITE: [usize; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

pub fn run_sequential(width: usize, height: usize, iterations: usize) {
    // relaksacioni parametar koji odredjuje viskoznost fluida
    let tau: f64 = 0.6;
    // glavna matrica distribucija: 9 pravaca po svakom cvoru (height x width)
    let mut f = Array3::<f64>::zeros((9, height, width));
    
    // definisanje geometrije prepreke (cilindricni objekat u kanalu)
    let mut obstacle = Array2::<bool>::from_elem((height, width), false);
    let (cx, cy, r) = (width / 4, height / 2, height / 10);
    for y in 0..height {
        for x in 0..width {
            // formula kruznice za definisanje granica objekta
            if ((x as isize - cx as isize).pow(2) + (y as isize - cy as isize).pow(2)) <= (r * r) as isize {
                obstacle[[y, x]] = true;
            }
        }
    }

    // onicijalizacija sistema u stanje lokalne ravnoteze (Equilibrium)
    for y in 0..height {
        for x in 0..width {
            let (ux, uy, rho) = (0.1, 0.0, 1.0); // pocetni protok s lijeva na desno
            let u_sq = ux*ux + uy*uy;
            for i in 0..9 {
                let u_dot_c = (CX[i] as f64) * ux + (CY[i] as f64) * uy;
                // LBM formula za ravnoteznu distribuciju (feq)
                f[[i, y, x]] = rho * W[i] * (1.0 + 3.0*u_dot_c + 4.5*u_dot_c*u_dot_c - 1.5*u_sq);
            }
        }
    }

    // glavna petlja simulacije
    for it in 0..iterations {
        // --- 1. STREAMING KORAK (advekcija) ---
        // oomjeranje cestica u susjedne cvorove na osnovu njihovih vektora brzina
        let mut f_new = f.clone(); // privremeni bafer (double buffering)
        for y in 0..height {
            for x in 0..width {
                for i in 0..9 {
                    // periodicni uslovi granica (fluid koji izadje desno, vraca se lijevo)
                    let next_y = (y as isize + CY[i]).rem_euclid(height as isize) as usize;
                    let next_x = (x as isize + CX[i]).rem_euclid(width as isize) as usize;
                    
                    if obstacle[[next_y, next_x]] {
                        // BOUNCE-BACK: Ako cestica udari u prepreku, vraca se u suprotnom smjeru
                        f_new[[OPPOSITE[i], y, x]] = f[[i, y, x]]; 
                    } else {
                        // standardno pomjeranje u slobodnom prostoru
                        f_new[[i, next_y, next_x]] = f[[i, y, x]];
                    }
                }
            }
        }
        f = f_new; // azuriranje stanja nakon pomjeranja

        // --- 2. COLLISION KORAK (relaksacija) ---
        // proracun sudara cestica i povratak ka ekvilibrijumu (BGK operator)
        for y in 0..height {
            for x in 0..width {
                if obstacle[[y, x]] { continue; } // preskacemo cvorove unutar prepreke
                
                // izracunavanje makroskopskih varijabli: gustina (rho) i brzina (ux, uy)
                let mut rho = 0.0;
                let (mut ux, mut uy) = (0.0, 0.0);
                for i in 0..9 {
                    rho += f[[i, y, x]];
                    ux += f[[i, y, x]] * (CX[i] as f64);
                    uy += f[[i, y, x]] * (CY[i] as f64);
                }
                ux /= rho; uy /= rho;
                
                let u_sq = ux*ux + uy*uy;
                for i in 0..9 {
                    let u_dot_c = (CX[i] as f64) * ux + (CY[i] as f64) * uy;
                    // proracun lokalnog ekvilibrijuma za trenutni cvor
                    let feq = rho * W[i] * (1.0 + 3.0*u_dot_c + 4.5*u_dot_c*u_dot_c - 1.5*u_sq);
                    // BGK jednacina sudara: f(t+1) = f(t) - (f(t) - feq) / tau
                    f[[i, y, x]] -= (f[[i, y, x]] - feq) / tau;
                }
            }
        }
        if it % 100 == 0 { println!("Iteracija {}", it); }
    }
    // cuvanje finalnog stanja za potrebe vizuelizacije
    save_state(&f, width, height);
}

/// pomocna funkcija za eksportovanje magnitude brzine u .dat fajl
fn save_state(f: &Array3<f64>, width: usize, height: usize) {
    let mut file = File::create("../data/rust_seq_state.dat").unwrap();
    for y in 0..height {
        for x in 0..width {
            let mut rho = 0.0;
            let (mut ux, mut uy) = (0.0, 0.0);
            for i in 0..9 {
                rho += f[[i, y, x]];
                ux += f[[i, y, x]] * (CX[i] as f64);
                uy += f[[i, y, x]] * (CY[i] as f64);
            }
            // magnituda brzine |u| = sqrt(ux^2 + uy^2)
            let mag = ((ux/rho).powi(2) + (uy/rho).powi(2)).sqrt();
            write!(file, "{} ", mag).unwrap();
        }
        write!(file, "\n").unwrap();
    }
}