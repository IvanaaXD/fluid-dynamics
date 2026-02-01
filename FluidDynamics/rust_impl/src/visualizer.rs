use plotters::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// generise heatmapu brzine fluida na osnovu izlaznih podataka simulacije
pub fn create_visualization(input_path: &str, output_path: &str, width: usize, height: usize) {
    // inicijalizacija bitmap backend-a sa dimenzijama koje odgovaraju mrezi simulacije
    let root = BitMapBackend::new(output_path, (width as u32, height as u32)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let file = File::open(input_path).expect("File with data not found in given folder");
    let reader = BufReader::new(file);
    let mut data = Vec::new();

    // 1. citanje podataka i normalizacija polja brzina
    // trazimo globalni maksimum kako bi boja bila relativna u odnosu na najbrzi protok u simulaciji
    let mut max_val = 0.001; // minimalna vrijednost radi sprijecavanja dijeljenja nulom (division by zero)
    for line in reader.lines() {
        let row: Vec<f64> = line.unwrap()
            .split_whitespace()
            .map(|v| v.parse().expect("Wrong data format in .dat file"))
            .collect();
        
        for &v in &row { 
            if v > max_val { max_val = v; } 
        }
        data.push(row);
    }

    // 2. mapiranje fizickih vrednosti u rgb spektar
    // prolazimo kroz svaki cvor (pixel) i crtamo pravougaonik 1x1
    for (y, row) in data.iter().enumerate() {
        for (x, &val) in row.iter().enumerate() {
            // mnormalizacija: 0.0 (plava - spora zona) do 1.0 (crvena - brza zona)
            let norm = (val / max_val * 255.0) as u8;
            
            // RGBColor
            // norm povecava crvenu komponentu, (255 - norm) smanjuje plavu
            root.draw(&Rectangle::new(
                [(x as i32, y as i32), (x as i32 + 1, y as i32 + 1)],
                RGBColor(norm, 0, 255 - norm).filled(),
            )).unwrap();
        }
    }
    
    root.present().expect("Error while generating PNG image");
}