
//Terminal User Interface
pub mod tui {
	
	pub fn print_vec<U: std::fmt::Display> (v: &Vec<U>) {
		let mut row = format!("{:>2}", v[0]);
		for k in 1..v.len() {
			row = format!("{} {:>2}", row, v[k]);
		}
		println!("{}", row);
	}

	pub fn print_2d_vec<U: std::fmt::Display>(v: &Vec<Vec<U>>) {
		for i in 0..v.len() {
			print_vec(&v[i]);
		}
	}

}

pub mod math_lib {

	pub trait StandardlyDistributable<T> {
		fn get_standard_distribution(v: Vec<T>) -> Self;
	}

	pub struct StandardDistribution {
		// pub data: Vec<f64>,
		pub mean: f64,
		pub deviation: f64
	}
	// pub struct StandardDistribution<f64> {
	// 	pub data: Vec<f64>,
	// 	pub mean: f64,
	// 	pub deviation: f64
	// }
	
	impl StandardlyDistributable<u32> for StandardDistribution {
		fn get_standard_distribution(v: Vec<u32>) -> Self {
			let n = f64::from(u32::try_from(v.len()).unwrap());
			let summation = f64::from(v.iter().sum::<u32>());
			let mean = summation/ n;

			let mut deviation = 0.0;
			for i in 0..v.len() {
				let x_f = f64::from(v[i]);
				deviation += x_f * x_f - 2.0 * mean * x_f + mean * mean;
			}

			deviation /= f64::from(u32::try_from(v.len() - 1).unwrap());
			return StandardDistribution { mean, deviation };
		}
	}

	pub fn test_sd() -> StandardDistribution {
		let line = (0..10).collect();
		let s_dist = StandardDistribution::get_standard_distribution(line);
		s_dist
	}

	impl StandardlyDistributable<f64> for StandardDistribution {
		fn get_standard_distribution(v: Vec<f64>) -> Self {
			let summation:f64 = v.iter().sum();
			let mean = summation / f64::from(u32::try_from(v.len()).unwrap());

			let mut deviation = 0.0;
			for i in 0..v.len() {
				let x_f = v[i];
				deviation += x_f * x_f - 2.0 * mean * x_f + mean * mean;
			}

			deviation /= f64::from(u32::try_from(v.len() - 1).unwrap());

			return StandardDistribution { mean, deviation };
		}
	}

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

	pub fn shuffle_vec (v: &mut Vec<u32>) {
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

	pub fn test_prime(sieve: &Vec<u32>, test_prime: &u32) -> bool {
		//A heuristic sqrt(n) search to then logarithmically find how much of the sieve is larger than the sqrt could be good.
		//This does not protect against sieve[-1] being greater than test_prime
		for i in 1..sieve.len() {
			match test_prime % sieve[i] { 
				0 => return false, 
				_ => continue
			}
		}
		true
	}

	pub fn prime_sieve(n: u32) -> Vec<u32> {
		if n < 2 { return vec!(); }
		
		let mut sieve = vec!(2);
		let mut test_val = 3;
		
		while test_val <= n {
			if test_prime(&sieve, &test_val) { 
				sieve.push(test_val); 
			}
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
			if moving_n <= 1 { return u128::from(fac); }
		}
	}
}

pub mod latin_square {
	use crate::math_lib::rand_usize;
	pub type UBitRep = u128;
	pub type UDisplayRep = u32;

	//TODO:
	//Verify that solve works.
	//Let solve be extended by polymorphism.

	pub struct LatinSquare<S> {
		pub square: Vec<Vec<UBitRep>>,
		pub heuristic: S
	}

	pub trait LatinSquareSolver {		
		fn heuristic_solve(&self, mut working_square: Vec<Vec<UBitRep>>) -> Vec<Vec<UBitRep>> {
			return working_square;
		}
	}

	// pub struct TowersSquare: LatinSquare {
	//	clues: Vec<UDisplayRep>,
	// 	square: Vec<Vec<UDisplayRep>>
	// }

	pub struct SudokuSolver;

	impl LatinSquareSolver for SudokuSolver {
		fn heuristic_solve(&self, mut working_square: Vec<Vec<UBitRep>>) -> Vec<Vec<UBitRep>> {
			let side_length = working_square.len();
			if side_length != 9 { panic!("Sudoku has been made that is not side length 9."); }

			for i in 0..side_length {

			}
			
			return working_square;
		}
	} 
	
	// pub struct KillerSudokuSquare: SudokuSquare {
	//	n: constant(9),
	// 	square: Vec<Vec<UDisplayRep>>,
	//  clues: Vec<KillerSudokuClue> //This will for sure need its own type.
	// }

	// pub struct KillerSudokuClue {
	// 	participants: Vec<Tuple<u32, u32>,
	//  condition: std::ops?, //idk. Plus or mult usually.
	//	clue: u32
	// }

	pub struct DefaultLatinSolver;

	impl LatinSquareSolver for DefaultLatinSolver {
		fn heuristic_solve(&self, working_square: Vec<Vec<UBitRep>>) -> Vec<Vec<UBitRep>> {
			return working_square;
		}
	}

	impl<S> LatinSquare<S> 
	where 
	S: LatinSquareSolver 
	{
		pub fn solve(&self) -> Vec<Vec<Vec<UBitRep>>> {
			let side_length = self.square.len();
			let mut solutions = Vec::new();
			let mut possible_solutions = Vec::new();
			
			let mut working_square = preprocess(&self.square);
			let mut previous_cum_cell_possibility = UBitRep::MAX;

			let mut analytics;
			super::tui::print_2d_vec(&working_square);
			let mut count = 0;
			loop {
				working_square = self.heuristic.heuristic_solve(working_square);
				analytics = analyse_square(&working_square);
				
				if analytics.cumulative_possibilities == 0 {
					super::tui::print_2d_vec(&working_square);
					if analytics.valid {
						solutions.push(working_square);
						println!("Pushing number {}", solutions.len());
					}

					
					println!("Popping number {}", possible_solutions.len());
					match possible_solutions.pop() {
						Some(s) => working_square = s,
						None => return solutions
					}
					previous_cum_cell_possibility = UBitRep::MAX;
					continue;
				}

				//If the loop wraps around with no progress, make a guess.
				if previous_cum_cell_possibility == analytics.cumulative_possibilities {
					let (i, j) = analytics.lowest_possibilities;
					
					for k in 0..side_length {
						let new_val = 1 << k & working_square[i][j];
						if new_val == 0 { continue; }

						let mut new_square = working_square.clone(); 
						if place(&mut new_square, &new_val, i, j) {
							possible_solutions.push(new_square);
						}
					}
					
					println!("Popping number {}", possible_solutions.len());
					match possible_solutions.pop() {
						Some(s) => working_square = s,
						None => return solutions
					}
					previous_cum_cell_possibility = UBitRep::MAX;
					continue;
				}

				previous_cum_cell_possibility = analytics.cumulative_possibilities;

				count += 1;
				if count == 100 { println!("Possible solutions: {}", possible_solutions.len()); return solutions; }
			}
		}
	}
	
	pub fn preprocess(square: &Vec<Vec<UBitRep>>) -> Vec<Vec<UBitRep>> {
		//1. Scan possibility map
		//a. create unconstrained val.
		let side_length = square.len();
		let unconstrained_value: UBitRep = (0..side_length).into_iter().map(|x| 1 << x).sum();

		//b. for any empty cell, insert the unconstrained value
		let mut working_square = (0..side_length).into_iter()
			.map(|_x| (0..side_length).into_iter()
				.map(|_x| unconstrained_value)
			.collect())
		.collect();

		for i in 0..side_length {
			for j in 0..side_length {
				//let this panic maybe?
				if square[i][j] == 0 || square[i][j] & (square[i][j] - 1) > 0 { continue; }
				place(&mut working_square, &square[i][j], i, j);
			}
		}

		return working_square;
	}

	struct LatinAnalytics {
		valid: bool,
		cumulative_possibilities: UBitRep,
		lowest_possibilities: (usize, usize)
	}

	impl Default for LatinAnalytics {
		fn default() -> LatinAnalytics {
			return LatinAnalytics {
				valid: false,
				cumulative_possibilities: 0,
				lowest_possibilities: (0, 0)
			}
		}
	}

	fn analyse_square(square: &Vec<Vec<UBitRep>>) -> LatinAnalytics {
		let side_length = square.len();

		let mut current_cum_cell_possibility = 0;
		let mut lowest_possibility = UBitRep::MAX;
		let mut lowest_possibility_coordinates = (0, 0);

		if square.len() == 0 { 
			return  LatinAnalytics::default();
		}

		for i in 0..side_length {
			for j in 0..side_length {
				let cell_possibilities = count_cell_possibilities(square[i][j]);
				if cell_possibilities == 0 { println!("Bad line."); return LatinAnalytics::default(); }
				if cell_possibilities == 1 { continue; }
				if cell_possibilities < lowest_possibility {
					lowest_possibility = cell_possibilities;
					lowest_possibility_coordinates = (i, j);
				}
				current_cum_cell_possibility += cell_possibilities;
			}
		}

		return LatinAnalytics{ 
			valid: true, 
			cumulative_possibilities: current_cum_cell_possibility, 
			lowest_possibilities: lowest_possibility_coordinates 
		};
	}

	pub fn gen_square(n: usize) -> Vec<Vec<UBitRep>> {
		loop {
			let square = generate_square_prob_collapse(n);
			if square.len() > 0 { return square; }
		}
	}

	pub fn place(working_square: &mut Vec<Vec<UBitRep>>, value: &UBitRep, i: usize, j: usize) -> bool {
		let side_length = working_square.len();
		if working_square[i][j] & value == 0 { return false; }//panic?
		
		working_square[i][j] = *value;
		
		for k in 0..side_length {
			if k != i && working_square[k][j] & value > 0 {
				working_square[k][j] &= !value;
				if working_square[k][j] == 0 {
					return false;
				}
				let possibility_check = working_square[k][j] & (working_square[k][j] - 1);
				if possibility_check == 0 {
					let new_value = working_square[k][j];
					place(working_square, &new_value, k, j);
				}	
			}

			if k != j && working_square[i][k] & value > 0 {
				working_square[i][k] &= !value;
				if working_square[i][k] == 0 {
					return false;
				}
				let possibility_check = working_square[i][k] & (working_square[i][k] - 1);
				if possibility_check == 0 {
					let new_value = working_square[i][k];
					place(working_square, &new_value, i, k);
				}	
			}
		}
		true
	}

	

	/**
	 * Removes n filled cells from the latin square, or all of the remaining cells
	 * if n is greater than the number of remaining cells.
	 */
	pub fn cull(mut working_square: Vec<Vec<UBitRep>>, n: usize) -> Vec<Vec<UBitRep>> {
		// let mut working_square = square.clone();
		let side_length = working_square.len();

		//This is probably quicker than infinite misses on generating random coordinates.
		let mut valid_matrix_indices: Vec<Vec<usize>> = (0..side_length).into_iter().map(
			|i| (0..side_length).into_iter().filter(
				|j| working_square[i][*j] != 0
			).collect()
		).collect();
		
		let mut valid_row_indices: Vec<usize> = (0..side_length).into_iter().filter(
			|i| valid_matrix_indices[*i].len() != 0
		).collect();
		
		let mut k = 0;
		while valid_row_indices.len() != 0 && k < n {
			let relative_i = rand_usize(k) % valid_row_indices.len();
			let i = valid_row_indices[relative_i];
			let relative_j = rand_usize(k) % valid_matrix_indices[i].len();
			let j = valid_matrix_indices[i][relative_j];

			working_square[i][j] = 0;

			valid_matrix_indices[i].remove(relative_j);
			if valid_matrix_indices[i].len() == 0 {
				valid_row_indices.remove(relative_i);
			}

			k += 1;
		}

		return working_square;
	}

	fn to_display(val: &UBitRep) -> UDisplayRep {
		let mut output = 0;
		if count_cell_possibilities(*val) > 1 { return 0; }
		if *val == 0 { return 0; }

		loop {
			if *val & 1 << output > 0 { return output + 1; }
			output += 1;
		}
	}

	pub fn display_format(v: Vec<Vec<UBitRep>>) -> Vec<Vec<UDisplayRep>> {
		let mut output = Vec::with_capacity(v.len());
		
		for line in v.iter() {
			output.push(
				line.iter()
					.map(|x| to_display(&x)).collect()
			);
		}
		
		output
	}
	
	fn count_cell_possibilities(mut target: UBitRep) -> UBitRep {
		// let mut target = val;
		let mut count = 0;
		
		//Kernighan's Algorithm
		while target != 0 {
			target = target & (target - 1);
			count += 1;
		}
		count
	}

	pub fn collapse_row_possibilities(row: &mut Vec<UBitRep>, target: usize) -> bool {
		for j in 0..row.len() {
			if j == target { continue; }

			if row[j] & row[target] != 0 {
				row[j] &= ! row[target];
				
				match count_cell_possibilities(row[j]) {
					0 => return false,
					1 => if !(collapse_row_possibilities(row, j)) { return false; },
					_ => continue
				}
			}
		}
		true
	}

	pub fn generate_valid_line(line: &mut Vec<UBitRep>, possibilities: &Vec<UBitRep>) -> bool {
		let n = line.len();
		let mut local_possibilities = possibilities.clone();

		//Inplace shuffle to reduce allocations.
		for i in 0..n {
			let mut misses: usize = 0;
			
			loop {
				let rand_index = i + rand_usize(i) % (n - i - misses);
				let temp = line[rand_index];

				if local_possibilities[i] & temp > 0 {
					line[rand_index] = line[i];
					line[i] = temp;
					local_possibilities[i] = temp;
					if !(collapse_row_possibilities(&mut local_possibilities, i)) { /*println!("NRRRRRRR");*/ return false; }
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

	pub fn generate_square_prob_collapse(n: usize) -> Vec<Vec<UBitRep>> {
		let prototype_line: Vec<UBitRep> = (0..n).into_iter().map(|x| 1 << x).collect();
		let side_length = prototype_line.len();
		
		let unconstrained_value = prototype_line.iter().sum();
		let mut possibilities: Vec<UBitRep> = (0..n).into_iter().map(|_x| unconstrained_value).collect();
		let mut square: Vec<Vec<UBitRep>> = Vec::with_capacity(side_length);

		for i in 0..(side_length - 1) {
			let mut new_line = prototype_line.clone();

			let mut count = 0;
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
}

pub mod latin_metrics {
	use std::time::Instant;
	// use super::math_lib::Unsigned;
	use super::latin_square::{ generate_square_prob_collapse, UBitRep };

	pub fn gen_square_prob_with_metrics(n: usize, count: &mut u32, millis: &mut u128) -> Vec<Vec<UBitRep>> {
		let timer = Instant::now();
		loop {
			*count += 1;
			let square: Vec<Vec<UBitRep>> = generate_square_prob_collapse(n);
			
			if square.len() > 0 {
				*millis = timer.elapsed().as_millis();
				return square;
			} else {
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use crate::math_lib::{ StandardlyDistributable, StandardDistribution };
	use super::latin_square::*;
	use super::math_lib;
	use super::latin_metrics::gen_square_prob_with_metrics;
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
		let mut line: Vec<u32> = (1..9).collect();
		let mut avgs: Vec<u32> = line.iter().map(|_x| 0).collect();
		
		let samples = 10000;
		for _n in 0..samples {
			math_lib::shuffle_vec(&mut line);
			avgs = (0..line.len()).map(|i| avgs[i] + line[i]).collect();
		}

		avgs = avgs.into_iter().map(|x| (10 * x + 1) / samples).collect();
		assert!(avgs.iter().max().unwrap() - avgs.iter().min().unwrap() <= 1);
	}
	
	#[test]
	fn test_prob_collapse() {
		let n = 60;
		let side_length = usize::try_from(n).unwrap();
		let samples = 10;
		for _i in 0..samples {
			let latin_square = display_format(gen_square(n));

			for i in 0..side_length {
				for j in 0..side_length {
					if i == j { continue; }
					assert!(latin_square[i][i] != latin_square [i][j]);
					assert!(latin_square[i][i] != latin_square [j][i]);
				}
			}
			// tui::print_2d_vec(&latin_square);
		}
	}

	#[test]
	#[ignore]
	fn wave_collapse_distribution_statistics() {
		let samples = 10;
		let size = 11;
		let vector_store: Vec<Vec<Vec<UDisplayRep>>> = (0..samples).map(|_x| display_format(gen_square(60))).collect();

		let mut mean_vector = Vec::with_capacity(size * size);
		let mut dev_vector = Vec::with_capacity(size * size);
		
		for i in 0..size {
			for j in 0..size {
				let line = (0..samples).map(|k| vector_store[k][j][i]).collect();

				let s_dist = StandardDistribution::get_standard_distribution(line);
				
				mean_vector.push(s_dist.mean);
				dev_vector.push(s_dist.deviation);
			}
		}
		
		let mean_dist = StandardDistribution::get_standard_distribution(mean_vector);
		let dev_dist = StandardDistribution::get_standard_distribution(dev_vector);
		println!("{} {}", mean_dist.mean, mean_dist.deviation);
		println!("{} {}", dev_dist.mean, dev_dist.deviation);

	}

    #[test]
	//could also try [bench] here.
    fn get_samples() {
		let total_time = Instant::now();
		let samples = 10;
		let min_size = 30;
		let max_size = 50;

		for n in min_size..max_size + 1 {
			let mut cum_count = 0;
			let mut cum_time_millis = 0;
			
			for _k in 0..samples {
				let result = gen_square_prob_with_metrics(n, &mut cum_count, &mut cum_time_millis);
				assert!(result.len() == usize::try_from(n).unwrap());
			}	

			println!("{} by {} Square:", n, n);
			println!("Average number of attempts:\t{}", cum_count / samples);
			println!("Average time:\t{}", cum_time_millis / u128::from(samples));
		}

		println!("Test suite ran for {} seconds.", total_time.elapsed().as_secs());
    }

	#[test]
	fn test_latin_culler() {
		let n = 6;
		let mut avgs: Vec<UDisplayRep> = (0..(n * n)).into_iter().map(|_x| 0).collect();
		//Magic number of samples at which point the average coalesces.;
		let samples = 5000;
		let cull_quantity = n * n / 2;

		super::tui::print_2d_vec(&display_format(cull(gen_square(n), cull_quantity)));
		
		for _i in 0..samples {

			let latin_square = gen_square(n);
			let display_square = display_format(cull(latin_square, cull_quantity));
			
			for k in 0..(n * n) {
				avgs[k] = avgs[k] + display_square[k / n][k % n];
			}
		}
			
		avgs = avgs.into_iter().map(|x| (10 * x + 1) / samples).collect();
		assert!(avgs.iter().max().unwrap() - avgs.iter().min().unwrap() <= 3);
	}

	#[test]
	fn test_latin_placer() {
		let mut latin_square = gen_square(5);
		let test_vec = latin_square[0].clone();

		for i in 1..latin_square.len() {
			latin_square[0][i] |= latin_square[0][0];
		}

		let i = 1;
		let j = 0;
		latin_square[0][0] |= latin_square[i][j];
		let val = latin_square[i][j];
		latin_square[i][j] = 31;

		place(&mut latin_square, &val, i, j);
		super::tui::print_2d_vec(&latin_square);

		assert!(latin_square[0] == test_vec);
	}

	#[test]
	fn test_latin_solver() {
		let n = 5;
		let mut latin_square = gen_square(n);

		println!("Initial square...");
		super::tui::print_2d_vec(&latin_square);
		println!();

		latin_square = cull(latin_square, 2 * n * n / 3);
		super::tui::print_2d_vec(&latin_square);
		println!();
		// let latin_squares = solve(latin_square);

		let latin = LatinSquare {
			square: latin_square,
			heuristic: DefaultLatinSolver
		};

		let latin_squares = latin.solve();
		if latin_squares.len() > 0 {
			super::tui::print_2d_vec(&latin_squares[0]);
		}
		println!("\nTEST_LATIN_SOLVER: {}", latin_squares.len());
	}
}
