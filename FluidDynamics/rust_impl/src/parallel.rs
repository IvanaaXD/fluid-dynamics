use ndarray::{Array2, Array3, Axis};
use std::time::Instant;
use ndarray::parallel::prelude::*;

const W: [f64; 9] = [4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0];
const CX: [f64; 9] = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0];
const CY: [f64; 9] = [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0];
const OPPOSITE: [usize; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

pub fn run_parallel(width: usize, height: usize, iterations: usize, num_threads: usize) {
    // konfiguracija globalnog Thread Pool-a za Rayon biblioteku
    let _ = rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global();
    let tau: f64 = 0.6;
    
    // inicijalizacija dva bafera za Double Buffering strategiju
    // ovim eliminisemo f.clone() iz petlje, sto drasticno smanjuje alokacije memorije
    let mut f = Array3::<f64>::zeros((9, height, width));
    let mut f_new = Array3::<f64>::zeros((9, height, width));

    // definisanje geometrijskih granica prepreke (cilindar)
    let mut obstacle = Array2::<bool>::from_elem((height, width), false);
    let (cx, cy, r) = (width / 4, height / 2, height / 10);
    for y in 0..height {
        for x in 0..width {
            if ((x as isize - cx as isize).pow(2) + (y as isize - cy as isize).pow(2)) <= (r * r) as isize {
                obstacle[[y, x]] = true;
            }
        }
    }

    // inicijalizacija polja brzina i gustine (Initial Equilibrium)
    for y in 0..height {
        for x in 0..width {
            let (ux, uy, rho) = (0.1, 0.0, 1.0);
            let u_sq = ux*ux + uy*uy;
            for i in 0..9 {
                let u_dot_c = CX[i] * ux + CY[i] * uy;
                f[[i, y, x]] = rho * W[i] * (1.0 + 3.0*u_dot_c + 4.5*u_dot_c*u_dot_c - 1.5*u_sq);
            }
        }
    }

    let start_time = Instant::now();
    for _ in 0..iterations {
        {
            let f_curr = &f;
            let obs = &obstacle;
            
            // LOOP FUSION: Spajamo streaming i collision korake u jedan paralelni prolaz
            // .axis_iter_mut(Axis(1)) dijeli matricu po redovima (Row-block partitioning)
            f_new.axis_iter_mut(Axis(1))
                .into_par_iter()
                .enumerate()
                .for_each(|(y, mut row_new)| {
                    for x in 0..width {
                        if obs[[y, x]] { continue; }

                        let mut rho = 0.0;
                        let (mut ux, mut uy) = (0.0, 0.0);

                        // PULL-MODEL STREAMING: umjesto da "guramo" vrijednosti u susjede (push), 
                        // svaka nit "povlaci" podatke koji joj trebaju iz prethodnog stanja (f_curr)
                        // ovo omogucava paralelno pisanje u f_new bez ikakvih Race Condition-a
                        for i in 0..9 {
                            let prev_y = (y as isize - CY[i] as isize).rem_euclid(height as isize) as usize;
                            let prev_x = (x as isize - CX[i] as isize).rem_euclid(width as isize) as usize;
                            
                            if obs[[prev_y, prev_x]] {
                                // realizacija Bounce-back uslova unutar pull modela
                                row_new[[i, x]] = f_curr[[OPPOSITE[i], y, x]];
                            } else {
                                // standardna advekcija (povlacenje vrijednosti iz susjednog cvora)
                                row_new[[i, x]] = f_curr[[i, prev_y, prev_x]];
                            }
                            
                            // akumulacija za makroskopske varijable tokom istog prolaza (Streaming + Collision info)
                            rho += row_new[[i, x]];
                            ux += row_new[[i, x]] * CX[i];
                            uy += row_new[[i, x]] * CY[i];
                        }

                        // COLLISION: relaksacija ka ekvilibrijumu (BGK operator)
                        ux /= rho; uy /= rho;
                        let u_sq = ux*ux + uy*uy;
                        for i in 0..9 {
                            let u_dot_c = CX[i] * ux + CY[i] * uy;
                            let feq = rho * W[i] * (1.0 + 3.0*u_dot_c + 4.5*u_dot_c*u_dot_c - 1.5*u_sq);
                            row_new[[i, x]] -= (row_new[[i, x]] - feq) / tau;
                        }
                    }
                });
        }
        // POINTER SWAP: umjesto kloniranja matrica (skupa operacija), samo mijenjamo 
        // reference na f i f_new u konstantnom vremenu O(1).
        std::mem::swap(&mut f, &mut f_new);
    }

    println!("Rust Rayon ({} niti) završena za: {:?}", num_threads, start_time.elapsed());
    save_output(&f, width, height);
}

/// funkcija za eksportovanje rezultata magnitude brzine u fajl
fn save_output(f: &Array3<f64>, width: usize, height: usize) {
    use std::io::Write;
    let mut file = std::fs::File::create("../data/rust_seq_state.dat").unwrap();
    for y in 0..height {
        for x in 0..width {
            let mut rho = 0.0;
            let (mut ux, mut uy) = (0.0, 0.0);
            for i in 0..9 {
                rho += f[[i, y, x]];
                ux += f[[i, y, x]] * CX[i];
                uy += f[[i, y, x]] * CY[i];
            }
            // izracunava magnitude vektora brzine |u|
            let mag = ((ux/rho).powi(2) + (uy/rho).powi(2)).sqrt();
            write!(file, "{} ", mag).unwrap();
        }
        write!(file, "\n").unwrap();
    }
}