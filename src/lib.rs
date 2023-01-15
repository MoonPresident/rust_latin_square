
pub mod tui {
	pub fn print_vec(v: &Vec<u32>) {
		let mut row = format!("{:>2}", v[0]);
		for k in 1..v.len() {
			row = format!("{} {:>2}", row, v[k]);
		}
		println!("{}", row);
	}

	pub fn print_2d_vec(v: &Vec<Vec<u32>>) {
		for i in 0..v.len() {
			print_vec(&v[i]);
		}
	}
}

pub mod math_lib {
	pub fn rand(val: &mut u32) -> i32 {
		unsafe {
			core::arch::x86_64::_rdrand32_step(val)
		}
	}
	
	pub fn rand_usize(index: usize) -> usize {
		let mut val = u32::try_from(index).unwrap();
		unsafe {
			core::arch::x86_64::_rdrand32_step(&mut val);
		}
		usize::try_from(val).unwrap()
	}

	pub fn prime_sieve(n: u32) -> Vec<u32> {
		if n < 2 { return vec!(); }
		
		let mut sieve = vec!(2);
		let mut test_val = 3;
		
		while test_val <= n {
			let mut include = true;

			for i in 1..sieve.len() {
				if test_val % sieve[i] == 0 { 
					include = false; 
					break; 
				}
			}
			
			if include { sieve.push(test_val); }
			test_val += 2; 
		}
		
		sieve
	}
	
	pub fn factorial(n: u32) -> u128 {
		if n == 0 || n == 1 { return 1; }
		if n == 2 { return 2; }
		if n < 2 {
			let mut fac: u32 = 2;
			for f in 3..n+1 { fac *= f; }
			return u128::from(fac);
		}
		
		let mut fac: u128 = 1;
		let sieve = prime_sieve(n);
		let mut moving_n = n;
		println!("");
		let mut power = 1;
		loop {
			let mut swing_fac:u128 = 1;
			
			for p in sieve.iter() {
				if p > &moving_n { break; }
				let mut q = moving_n;
				let mut f = 1;
				
				while q > 1 {
					q /= p;
					if q & 1 > 0 { f *= p; }
				}
				
				swing_fac *= u128::from(f);
			}
			
			fac *= swing_fac.pow(power);
			power = power << 1;
			
			moving_n = moving_n >> 1;
			if moving_n == 1 { return u128::from(fac); }
		}
	}
}

pub mod latin_square {
	use crate::math_lib::rand_usize;
	use super::tui::print_vec;
	use std::time::Instant;

	pub fn shuffle(v: &mut Vec<u32>) {
		//For each index, grab a random value stored between that index and the final index
		//(inclusive of current and final indices) then swap that with the current index.
		//If the current index is chosen, it remains the same. (Swaps with itself).
		//Don't add an if statement to check is i == k, it just add cycles in most cases.
		for i in 0..v.len() - 1 {
			let k = i + (rand_usize(i) % (v.len() - i));
			let temp = v[i];
			v[i] = v[k];
			v[k] = temp;
		}
	}
	
	/**
	 * Number of combinations in a line is n! (factorial)
	 * Number of combinations is n! ^ x
	 * Number of combinations with overlap:
	 * 		1: 1/n
	 * 		2: 1/(n * (n - 1))
	 * 		3: 1/(n * (n - 1) * (n - 2))
	 * 		c: 1/(n! - (n - c)!)
	 * 		n: 1/n!
	 * Sum: 1 / (n * n! - (n - 1)! - (n - 2)! ... - !
	 */
	pub fn validate_row(square: &Vec<Vec<u32>>, row: &Vec<u32>, from: usize) -> bool {
		for j in 0..(from) {
			for k in 0..row.len() {
				if row[k] == square[j][k + from] { 
					return false;
				}
			}
		}	
		true
	}
	
	pub fn validate_col(square: &Vec<Vec<u32>>, col: &Vec<u32>, from: usize) -> bool {
		for j in 0..(from) {
			for k in 0..col.len() {
				if col[k] == square[k + from][j] { 
					return false;
				}
			}
		}
		true
	}
	
	fn possible_number_of_values(val: u32) -> u32 {
		let mut target = val;
		let mut count = 0;
		
		//Kernighan's Algorithm
		while target != 0 {
			target = target & (target - 1);
			count += 1;
		}
			// 	if binary_comparitor & val > 0 { count += 1; }
		// 	if binary_comparitor & 1 << 31 > 0 { break; }
		// 	binary_comparitor = binary_comparitor << 1;
		// }
		
		count
	}

	// pub fn possible_row_values(line: Vec<u32>, exclusion_list: Vec<u32>) -> Vec<u32> {
	// 	Vec::new()
	// }

	// pub fn possible_col_values(line: Vec<u32>, exclusion_list: Vec<u32>, col: u32) -> Vec<u32> {
	// 	Vec::new()
	// }
	pub fn collapse_row_possibilities(row: &mut Vec<u32>, target: usize) -> bool {
		for j in 0..row.len() {
			if j == target { continue; }

			if row[j] & row[target] != 0 {
				row[j] &= ! row[target];
				
				match possible_number_of_values(row[j]) {
					0 => return false,
					1 => if !(collapse_row_possibilities(row, j)) { return false; },
					_ => continue
				}
			}
		}
		true
	}

	pub fn generate_valid_line(line: &mut Vec<u32>, possibilities: &Vec<u32>) -> bool {
		let n = line.len();
		let mut local_possibilities = possibilities.clone();

		for i in 0..n {
			let mut misses: usize = 0;
			
			loop {
				let rand_index = i + rand_usize(i) % (n - i - misses);
				let temp = line[rand_index];

				if local_possibilities[i] & temp > 0 {
					line[rand_index] = line[i];
					line[i] = temp;
					local_possibilities[i] = temp;
					if !(collapse_row_possibilities(&mut local_possibilities, i)) { println!("NRRRRRRR"); return false; }
					break;
				} else {
					line[rand_index] = line[n - misses - 1];
					line[n - misses - 1] = temp;
					misses += 1;
				}
			}
		}

		true
	}

	pub fn generate_square_prob_collapse(n: u32) -> Vec<Vec<u32>> {
		let prototype_line: Vec<u32> = (0..n).into_iter().map(|x| 1 << x).collect();
		let side_length = prototype_line.len();
		
		let unconstrained_value = prototype_line.clone().into_iter().sum();
		
		let mut possibilities: Vec<u32> = (0..n).into_iter().map(|_x| unconstrained_value).collect();
		let mut square: Vec<Vec<u32>> = Vec::with_capacity(side_length);

		for i in 0..(side_length - 1) {
			let mut new_line = prototype_line.clone();

			let mut count: u32= 0;
			loop {
				if generate_valid_line(&mut new_line, &possibilities) { break; }
				count += 1;
				if count == 100 { return Vec::new(); }
			}
			
			square.push(new_line);

			for j in 0..side_length {
				possibilities[j] &= !square[i][j];
			}
		}

		square.push(possibilities);
		square
	}
	
	// pub fn generate_square_prob_collapse(n: u32) -> Vec<Vec<u32>> {
	// 	let prototype_line: Vec<u32> = (0..n).into_iter().map(|x| 1 << x).collect();
	// 	let side_length = prototype_line.len();
	// 	let mut square: Vec<Vec<u32>> = Vec::with_capacity(side_length);

	// 	let mut unconstrained_value = 0;
	// 	for value in &prototype_line { unconstrained_value |= value; }
		
	// 	let empty_line: Vec<u32> = (0..n).into_iter().map(|_x: u32| 0).collect();
		
	// 	square.push(prototype_line.clone());
	// 	shuffle(&mut square[0]);

	// 	let mut col = prototype_line.clone().into_iter().filter(|v| *v != square[0][0]).collect();
	// 	shuffle(&mut col);

	// 	for j in 1..side_length { 
	// 		square.push(empty_line.clone()); 
	// 		square[j][0] = col[j - 1]; 
	// 	}

	// 	let mut square_dofs: Vec<Vec<u32>> = Vec::with_capacity(side_length);
	// 	for _i in 0..side_length { square_dofs.push(empty_line.clone()); }

	// 	for i in 1..(1 + side_length / 2) {
			
	// 		let mut col = prototype_line.clone();
	// 		for j in 0..i { col.remove(col.binary_search(&square[j][i]).unwrap()); }
			
	// 		loop {
	// 			shuffle(&mut col);
	// 			if validate_col(&square, &col, i) { break; }
	// 		}
			
	// 		for j in i..side_length { square[j][i] = col[j - i]; }
			
	// 		let mut row = prototype_line.clone();
	// 		for j in 0..i { row.remove(row.binary_search(&square[i][j]).unwrap()); }
			
	// 		loop {
	// 			shuffle(&mut row);
	// 			if validate_row(&square, &row, i) { break; }
	// 		}
			
	// 		for j in i..side_length { square[i][j] = row[j - i]; }
	// 	} 
		
	// 	//Filling the square halfway is probably quicker with the above method. More tests later on.
	// 	let start = 1 + side_length / 2;
	// 	for i in (1 + side_length / 2)..side_length {
	// 		for j in (1 + side_length / 2)..side_length {
	// 			square[i][j] = unconstrained_value;
	// 			println!("\nCoords {} and {}...", i, j);
	// 			for m in 0..(start) {println!("Comparing {} to {}.", square[i][j], square[m][j]);
	// 				square[i][j] &= !square[m][j];
	// 			}
	// 			for m in 0..(start) {println!("Comparing {} to {}.", square[i][j], square[i][m]);
	// 				square[i][j] &= !square[i][m];
	// 			}
				
	// 			//get DOF.
	// 			square_dofs[i][j] = possible_number_of_values(square[i][j]);
	// 			if square_dofs[i][j] == 0  { println!("BadExit on 0 at {i}, {j}."); return square; }
	// 			if square_dofs[i][j] == 1  { 
	// 				for m in start..(side_length) {println!("Comparing {} to {}.", square[i][j], square[m][j]);
	// 					if m == i { continue; }
	// 					square[m][j] &= !square[i][j];
	// 				}
	// 				for m in start..(side_length) {println!("Comparing {} to {}.", square[i][j], square[i][m]);
	// 					if m == i { continue; }
	// 					square[i][m] &= !square[i][j];
	// 				}
	// 			}
	// 		}
	// 	}
		
	// 	// tui::print_2d_vec(&square);
		
	// 	square
	// }
	
	
	
	
	pub fn generate_square_shuffle(n: u32) -> Vec<Vec<u32>> {
		let prototype_line: Vec<u32> = (1..n + 1).collect();
		let side_length = prototype_line.len();
		let mut square: Vec<Vec<u32>> = Vec::with_capacity(side_length);
		let mut empty_line: Vec<u32> = Vec::with_capacity(side_length);
		for _i in 0..side_length { empty_line.push(0); }
		
		let mut row = prototype_line.clone();
		shuffle(&mut row);
		square.push(row);
		for _i in 1..side_length { square.push(empty_line.clone()); }
		
		for i in 1..side_length {
			let mut col = prototype_line.clone();
			
			for j in 0..i { col.remove(col.binary_search(&square[j][i - 1]).unwrap()); }
			
			let mut count = 0;
			while count < 1000 {
				shuffle(&mut col);
				if validate_col(&square, &col, i) { break; }
				count = count + 1;
			}
			if count == 1000 { return Vec::new(); }
			
			
			for j in i..side_length { square[j][i - 1] = col[j - i]; }
			
			let mut row = prototype_line.clone();
			for j in 0..i { row.remove(row.binary_search(&square[i][j]).unwrap()); }
			
			let mut count = 0;
			while count < 1000 {
				shuffle(&mut row);
				if validate_row(&square, &row, i) { break; }
				count = count + 1;
			}
			
			if count == 1000 { return Vec::new(); }
			
			for j in i..side_length { square[i][j] = row[j - i]; }
		} 

		square
	}
	
	pub fn gen_square(n: u32) -> Vec<Vec<u32>> {
		loop {
			let square = generate_square_shuffle(n);
			if square.len() > 0 { return square; }
		}
	}
	
	pub fn gen_square_with_metrics(n: u32, count: &mut u32, millis: &mut u128) -> Vec<Vec<u32>> {
		let timer = Instant::now();
		loop {
			*count = *count + 1;
			let square: Vec<Vec<u32>> = generate_square_shuffle(n);
			
			if square.len() > 0 {
				*millis = timer.elapsed().as_millis();
				return square;
			}
		}
	}

	pub fn gen_square_prob_with_metrics(n: u32, count: &mut u32, millis: &mut u128) -> Vec<Vec<u32>> {
		let timer = Instant::now();
		loop {
			*count = *count + 1;
			let square: Vec<Vec<u32>> = generate_square_prob_collapse(n);
			
			if square.len() > 0 {
				*millis = timer.elapsed().as_millis();
				return square;
			}
		}
	}
	 
}

#[cfg(test)]
mod tests {
	use super::{latin_square, math_lib, tui};
	use std::time::Instant;
	
	#[test]
	#[ignore]
	fn test_prime_sieve() {
		assert_eq!(math_lib::prime_sieve(2), vec!(2));
		assert_eq!(math_lib::prime_sieve(3), vec!(2, 3));
		assert_eq!(math_lib::prime_sieve(4), vec!(2, 3));
		assert_eq!(math_lib::prime_sieve(6), vec!(2, 3, 5));
		assert_eq!(math_lib::prime_sieve(15), vec!(2, 3, 5, 7, 11, 13));
	}
	
	#[test]
	#[ignore]
	fn test_factorial() {
		let facs = vec!(1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600);
		for i in 0..facs.len() {
			assert_eq!(math_lib::factorial(u32::try_from(i).unwrap()), facs[i]);
		}
	}
	
	#[test]
	#[ignore]
	fn test_shuffle_randomness() {
		let prototype_line: Vec<u32> = (1..9).collect();
		let mut avgs: Vec<u32> = Vec::with_capacity(prototype_line.len());
		for _i in 0..prototype_line.len() { avgs.push(0); }
		
		let samples = 10000;
		for _n in 0..samples {
			let mut temp = prototype_line.clone();
			latin_square::shuffle(&mut temp);
			
			for i in 0..prototype_line.len() { avgs[i] = avgs[i] + temp[i]; }
		}
		
		for i in 0..prototype_line.len() { println!("{}: {}", i, 10 * (avgs[i] + 1) / samples); }
		
	}
	
	#[test]
	fn test_prob_collapse() {
		for i in 0..100 {
			let latin = latin_square::generate_square_prob_collapse(25);
			println!();
			tui::print_2d_vec(&latin);
		}
	}

    #[test]
	//could also try [bench] here.
    fn get_samples() {
		let total_time = Instant::now();
		let samples = 100;
		let min_size = 8;
		let max_size = 11;
		for n in min_size..max_size + 1 {				
			let mut cum_shuffle_count = 0;
			let mut cum_shuffle_time = 0;
			let mut cum_wave_collapse_count = 0;
			let mut cum_wave_collapse_time = 0;
			for _k in 0..samples {
				let mut count = 0;
				let mut millis = 0;
				let result = latin_square::gen_square_with_metrics(n, &mut count, &mut millis);
				assert!(result.len() == usize::try_from(n).unwrap());
				
				cum_shuffle_count = cum_shuffle_count + count;
				cum_shuffle_time = cum_shuffle_time + millis;
			}	
			
			for _k in 0..samples {
				let mut count = 0;
				let mut millis = 0;
				let result = latin_square::gen_square_prob_with_metrics(n, &mut count, &mut millis);
				assert!(result.len() == usize::try_from(n).unwrap());
				
				cum_wave_collapse_count = cum_wave_collapse_count + count;
				cum_wave_collapse_time = cum_wave_collapse_time + millis;
			}	

			println!("{} by {} Square:", n, n);
			println!("Average number of attempts:\t{} \t- \t{}", cum_shuffle_count / samples, cum_shuffle_count / samples);
			println!("Average time:\t{} \t- \t{}", cum_shuffle_time / u128::from(samples), cum_wave_collapse_time / u128::from(samples));
		}
		

		println!("Test suite ran for {} seconds.", total_time.elapsed().as_secs());
    }
}
