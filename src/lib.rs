
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

pub mod latin_square {
	use moon_math::moon_math::rand_usize;
	pub type UBitRep = u128;
	pub type UDisplayRep = u32;

	//TODO:
	//Let solve be extended by polymorphism.

	pub struct LatinSquare<S> {
		pub square: Vec<Vec<UBitRep>>,
		pub rules: S
	}

	pub trait LatinSquareSolver {
		fn generate(&self, n: usize) -> Vec<Vec<UBitRep>> {
			return gen_square(n);
		}	

		fn heuristic_solve(&self, working_square: Vec<Vec<UBitRep>>) -> Vec<Vec<UBitRep>> {
			return working_square;
		}

		fn heuristic_place(&self, working_square: &Vec<Vec<UBitRep>>, value: &UBitRep, i: usize, j: usize) -> bool {
			return true;
		}
	}

	// pub struct TowersSquare: LatinSquare {
	//	clues: Vec<UDisplayRep>,
	// 	square: Vec<Vec<UDisplayRep>>
	// }

	pub struct SudokuSolver;

	// impl LatinSquareSolver for SudokuSolver {
	// 	// fn heuristic_constraint(&self, mut working_square)
	// 	fn heuristic_solve(&self, mut working_square: Vec<Vec<UBitRep>>) -> Vec<Vec<UBitRep>> {
	// 		let side_length = working_square.len();
	// 		if side_length != 9 { panic!("Sudoku has been made that is not side length 9."); }

	// 		for subsquare in 0..side_length {
	// 			let mut col_start = subsquare % 3;
	// 			col_start *= 3;
	// 			let mut row_start = subsquare / 3;
	// 			row_start *= 3;
	// 			let mut row_possibility_array: Vec<Vec<usize>> = (0..side_length).into_iter().map(|x| vec![0, 0, 0]).collect();
	// 			let mut col_possibility_array: Vec<Vec<usize>> = (0..side_length).into_iter().map(|x| vec![0, 0, 0]).collect();
				
	// 			for i in 0..3 {
	// 				for j in 0..3 {
	// 					let target = working_square[row_start + i][col_start + j];
	// 					if count_cell_possibilities(target) > 0 {

	// 						for k in 0..side_length {
	// 							if 1 << k & target > 0 {
	// 								row_possibility_array[i][k] = 1;
	// 								col_possibility_array[j][k] = 1;
	// 							}
	// 						}

	// 					}
	// 				}
	// 			}

	// 			for i in 0..side_length {
	// 				if row_possibility_array[i].iter().sum::<usize>() == 1 {
	// 					let mut k = 0;
	// 					while(row_possibility_array[i][k] == 0) { k += 1; }

	// 					// for col in 0..col_start {
	// 					// 	working_square
	// 					// }

	// 					//clean this value from the rest of this row for the working square.
	// 				}

	// 				if col_possibility_array[i].iter().sum::<usize>() == 1 {
	// 					//clean this value from the rest of this row for the working square.
	// 				}
	// 			}
	// 		}
			
	// 		return working_square;
	// 	}

	// 	fn heuristic_place(&self, mut working_square: &Vec<Vec<UBitRep>>, value: &UBitRep, i: usize, j: usize) -> bool {
	// 		//for now assume 9
	// 		let side_length = working_square.len();
	// 		let mut cell_row = i / 3;
	// 		cell_row *= 3;
	// 		let mut cell_col = j / 3;
	// 		cell_col *= 3;

	// 		let mut coords: Vec<(usize, usize)> = Vec::with_capacity(side_length - 1);

	// 		for row in cell_row..cell_row + 3 {
	// 			for col in cell_col..cell_col + 3 {
	// 				if row == i && col == j { continue; }
	// 				coords.push((row, col));
	// 			}
	// 		}

	// 		for coord in coords {
	// 			if working_square[coord.0][coord.1] & value > 0 {
	// 				working_square[coord.0][coord.1] &= !value;

	// 				let target = working_square[coord.0][coord.1];
	// 				if target == 0 {
	// 					return false;
	// 				}
	// 				let possibility_check = target & (target - 1);
	// 				if possibility_check == 0 {
	// 					let new_value = target;
	// 					self.heuristic_place(&mut working_square, &new_value, coord.0, coord.1);
	// 				}	
	// 			}
	// 		}
	// 		true
	// 	}
	// }
	
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
			let mut partial_solutions = Vec::new();
			
			let mut analytics = LatinAnalytics::default();
			let mut count = 0;
			
			let mut working_square = Self::preprocess(&self, &self.square);
			loop {
				loop {
					working_square = self.rules.heuristic_solve(working_square);
					analytics = analyse_square(&working_square, analytics);
					if analytics.cumulative_possibilities == 0 || !analytics.analytics_updated { 
						break; 
					}
				}
				
				if analytics.cumulative_possibilities == 0 && analytics.valid {
					solutions.push(working_square);
				}  else if !analytics.analytics_updated {
					//If the loop wraps around with no progress, make a guess.
					let (i, j) = analytics.lowest_possibilities;
					let cell = working_square[i][j];
					let branching_values: Vec<UBitRep> = (0..side_length).map(|x| 1 << x & cell).filter(|x| *x > 0).collect();
					for new_val in branching_values {
						let mut new_square = working_square.clone(); 
						if self.place(&mut new_square, &new_val, i, j) {
							partial_solutions.push(new_square);
						}
					}
				}

				match partial_solutions.pop() {
					Some(s) => working_square = s,
					None => return solutions
				}

				count += 1;
				if count == 100 { println!("Partial solutions: {}", partial_solutions.len()); return solutions; }
			}
		}

		pub fn place(&self, working_square: &mut Vec<Vec<UBitRep>>, value: &UBitRep, i: usize, j: usize) -> bool {
			let side_length = working_square.len();
			if working_square[i][j] & value == 0 { return false; }//panic?
			
			working_square[i][j] = *value;
			
			let mut coords: Vec<(usize, usize)> = (0..side_length).filter(|k| *k != i).map(|k| (k, j)).collect();
			let mut col_coords: Vec<(usize, usize)> = (0..side_length).filter(|k| *k != j).map(|k| (i, k)).collect();
			coords.append(&mut col_coords);

			for coord in coords {
				if working_square[coord.0][coord.1] & value > 0 {
					working_square[coord.0][coord.1] &= !value;

					let target = working_square[coord.0][coord.1];
					if target == 0 {
						return false;
					}
					let possibility_check = target & (target - 1);
					if possibility_check == 0 {
						let new_value = target;
						self.place(working_square, &new_value, coord.0, coord.1);
						self.rules.heuristic_place(working_square, &new_value, coord.0, coord.1);
					}	
				}
			}
			true
		}
		
		pub fn preprocess(&self, square: &Vec<Vec<UBitRep>>) -> Vec<Vec<UBitRep>> {
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
					self.place(&mut working_square, &square[i][j], i, j);
				}
			}

			return working_square;
		}
	}
	
	

	struct LatinAnalytics {
		valid: bool,
		cumulative_possibilities: UBitRep,
		lowest_possibilities: (usize, usize),
		analytics_updated: bool
	}

	impl Default for LatinAnalytics {
		fn default() -> LatinAnalytics {
			return LatinAnalytics {
				valid: false,
				cumulative_possibilities: 0,
				lowest_possibilities: (0, 0),
				analytics_updated: false,
			}
		}
	}

	fn analyse_square(square: &Vec<Vec<UBitRep>>, prev_analytics: LatinAnalytics) -> LatinAnalytics {
		let side_length = square.len();

		let mut cumulative_possibilities = 0;
		let mut lowest_possibility = UBitRep::MAX;
		let mut lowest_possibility_coordinates = (0, 0);

		if square.len() == 0 { 
			return  LatinAnalytics::default();
		}

		for i in 0..side_length {
			for j in 0..side_length {
				let cell_possibilities = count_cell_possibilities(square[i][j]);
				if cell_possibilities == 0 { return LatinAnalytics::default(); }
				if cell_possibilities == 1 { continue; }
				if cell_possibilities < lowest_possibility {
					lowest_possibility = cell_possibilities;
					lowest_possibility_coordinates = (i, j);
				}
				cumulative_possibilities += cell_possibilities;
			}
		}

		return LatinAnalytics{ 
			valid: true, 
			cumulative_possibilities: cumulative_possibilities, 
			lowest_possibilities: lowest_possibility_coordinates,
			analytics_updated: prev_analytics.cumulative_possibilities != cumulative_possibilities
		};
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

	pub fn gen_square(n: usize) -> Vec<Vec<UBitRep>> {
		let side_length = n;
		let mut square: Vec<Vec<UBitRep>> = Vec::with_capacity(side_length);
		for i in 0..side_length {
			let mut new_line: Vec<UBitRep> = (i..n).into_iter().map(|x| 1 << x).collect();
			let mut new_line_back: Vec<UBitRep> = (0..i).into_iter().map(|x| 1 << x).collect();
			new_line.append(&mut new_line_back);
			square.push(new_line);
		}

		for i in 0..side_length - 1 {
			let target = side_length - i - 1;
			let mut source = rand_usize(target) % (target + 1);

			if target != source {
				let temp = square[target].clone();
				square[target] = square[source].clone();
				square[source] = temp;
			}

			source = rand_usize(target) % (target + 1);

			if target != source {
				for j in 0..side_length {
					let temp = square[j][target];
					square[j][target] = square[j][source];
					square[j][source] = temp;
				}
			}

		}

		square
	}
}

#[cfg(test)]
mod tests {
	use moon_stats::moon_stats::{ StandardlyDistributable, StandardDistribution };
	use super::latin_square::*;
	use moon_math::moon_math;
	use std::time::Instant;
	
	#[test]
	#[ignore]
	fn test_shuffle_randomness() {
		let mut line: Vec<u32> = (1..9).collect();
		let mut avgs: Vec<u32> = line.iter().map(|_x| 0).collect();
		
		let samples = 10000;
		for _n in 0..samples {
			moon_math::shuffle_vec(&mut line);
			avgs = (0..line.len()).map(|i| avgs[i] + line[i]).collect();
		}

		avgs = avgs.into_iter().map(|x| (10 * x + 1) / samples).collect();
		assert!(avgs.iter().max().unwrap() - avgs.iter().min().unwrap() <= 1);
	}

	fn validate_latin_square<T: std::cmp::PartialEq>(latin_square: &Vec<Vec<T>>){
		let side_length = latin_square.len();
		for i in 0..side_length {
			for j in 0..side_length {
				if i == j { continue; }
				assert!(latin_square[i][i] != latin_square [i][j]);
				assert!(latin_square[i][i] != latin_square [j][i]);
			}
		}
	}
	
	#[test]
	fn test_prob_collapse() {
		let n = 20;
		let samples = 10;
		for _i in 0..samples {
			let latin_square = display_format(gen_square(n));
			super::tui::print_2d_vec(&latin_square);
			validate_latin_square(&latin_square);
		}
	}

	#[test]
	#[ignore]
	fn wave_collapse_distribution_statistics() {
		let samples = 100;
		let size = 11;
		let vector_store: Vec<Vec<Vec<UDisplayRep>>> = (0..samples).map(|_x| display_format(gen_square(60))).collect();

		// for sample in vector_store.iter() { super::tui::print_2d_vec(&sample); println!(); }
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
		let samples: u32 = 100;
		let min_size = 40;
		let max_size = 60;

		for n in min_size..max_size + 1 {
			let mut cum_time_millis = total_time.elapsed().as_millis();
			
			let _results: Vec<Vec<Vec<UBitRep>>> = (0..samples).into_iter().map(|_x| gen_square(n)).collect();

			cum_time_millis = total_time.elapsed().as_millis() - cum_time_millis;
			println!("N: {:>3} | {:>5} |", n, cum_time_millis);
		}

		println!("Test suite ran for {} seconds.", total_time.elapsed().as_secs());
    }
	
	#[test]
	#[should_panic]
	fn oversize_square() {
		gen_square(20000);
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
		
		let latin = LatinSquare {
			square: latin_square.clone(),
			rules: DefaultLatinSolver
		};


		super::tui::print_2d_vec(&latin.square);
		latin.place(&mut latin_square, &val, i, j);
		super::tui::print_2d_vec(&latin.square);

		assert!(latin_square[0] == test_vec);
	}

	#[test]
	fn test_latin_solver() {
		let n = 5;
		let mut latin_square = gen_square(n);

		println!("Initial square...");
		super::tui::print_2d_vec(&latin_square);
		println!();

		latin_square = cull(latin_square, 3 * n * n / 5);
		super::tui::print_2d_vec(&latin_square);
		println!();
		// let latin_squares = solve(latin_square);

		let latin = LatinSquare {
			square: latin_square,
			rules: DefaultLatinSolver
		};

		let latin_squares = latin.solve();
		if latin_squares.len() > 0 {
			super::tui::print_2d_vec(&latin_squares[0]);
		}
		println!("\nTEST_LATIN_SOLVER: {}", latin_squares.len());
	}
}
