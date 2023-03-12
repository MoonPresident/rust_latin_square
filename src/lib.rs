
//Terminal User Interface
pub mod tui {
	
	pub fn print_vec<U: std::fmt::Display> (v: &Vec<U>) {
		let mut row = format!("{:>3}", v[0]);
		for k in 1..v.len() {
			row = format!("{} {:>3}", row, v[k]);
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

	type Vec2D<T> = Vec<Vec<T>>;
	pub trait LatinSquare {
		fn place(&mut self, i: usize, j: usize, value: UDisplayRep);
		fn get_square(&self) -> Vec2D<UDisplayRep>;
		fn heuristic_solve(&self, square: Vec2D<UBitRep>) -> Vec2D<UBitRep> { return square; }
		#[allow(unused_variables)]
		fn heuristic_place_and_update(&self, square: &mut Vec2D<UBitRep>, new_value: &UBitRep, coord1: usize, coord2: usize) -> bool { return true; }

		fn solve(&self) -> Vec<Vec<Vec<UBitRep>>> {
			let square = self.get_square();
			let square = bit_format(square);
			let side_length = square.len();
			let mut solutions = Vec::new();
			let mut partial_solutions = Vec::new();
			
			let mut analytics = LatinAnalytics::default();
			let mut count = 0;

			let mut working_square = Self::preprocess(&self, &square);
			loop {
				loop {
					working_square = self.heuristic_solve(working_square);
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
						if self.place_and_update(&mut new_square, &new_val, i, j) {
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

		fn place_and_update(&self, working_square: &mut Vec<Vec<UBitRep>>, value: &UBitRep, i: usize, j: usize) -> bool {
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
						self.place_and_update(working_square, &new_value, coord.0, coord.1);
						self.heuristic_place_and_update(working_square, &new_value, coord.0, coord.1);
					}	
				}
			}
			true
		}
		
		fn preprocess(&self, square: &Vec<Vec<UBitRep>>) -> Vec<Vec<UBitRep>> {
			let side_length = square.len();
			let unconstrained_value: UBitRep = (0..side_length).map(|x| 1 << x).sum();

			let mut working_square = (0..side_length).map(
				|_x| (0..side_length).map(
					|_x| unconstrained_value
				).collect()
			).collect();

			for i in 0..side_length {
				for j in 0..side_length {
					if square[i][j] == 0 || square[i][j] & (square[i][j] - 1) > 0 { continue; }
					self.place_and_update(&mut working_square, &square[i][j], i, j);
					self.heuristic_place_and_update(&mut working_square, &square[i][j], i, j);
				}
			}

			return working_square;
		}
	}

	pub struct DefaultLatinSquare {
		pub square: Vec2D<UDisplayRep>,
		// pub analytics: LatinAnalytics
	}

	impl LatinSquare for DefaultLatinSquare {
		fn get_square(&self) -> Vec2D<UDisplayRep> { return self.square.clone(); }

		fn place(&mut self, i: usize, j: usize, value: UDisplayRep) { self.square[i][j] = value; }
	}

	// pub struct TowersSquare: LatinSquare {
	//	clues: Vec<UDisplayRep>,
	// 	square: Vec<Vec<UDisplayRep>>
	// }

	pub struct Sudoku {
		pub square: Vec2D<UDisplayRep>,
		// pub analytics: LatinAnalytics
	}

	impl LatinSquare for Sudoku {
		fn get_square(&self) -> Vec2D<UDisplayRep> { return self.square.clone(); }
		fn place(&mut self, i: usize, j: usize, value: UDisplayRep) { self.square[i][j] = value; }

		// fn heuristic_constraint(&self, mut working_square)
		fn heuristic_solve(&self, mut working_square: Vec<Vec<UBitRep>>) -> Vec<Vec<UBitRep>> {
			return working_square;
			let side_length = working_square.len();
			if side_length != 9 { panic!("Sudoku has been made that is not side length 9."); }

			for subsquare in 0..side_length {
				let mut col_start = subsquare % 3;
				col_start *= 3;
				let mut row_start = subsquare / 3;
				row_start *= 3;
				let mut row_possibility_array: Vec<Vec<usize>> = (0..side_length).into_iter().map(|_x| vec![0, 0, 0]).collect();
				let mut col_possibility_array: Vec<Vec<usize>> = (0..side_length).into_iter().map(|_x| vec![0, 0, 0]).collect();
				
				let targets: Vec<(usize, usize)> = (0..9).into_iter().map(
					|i| (i / 3, i % 3)
				).filter(
					|&t| (count_cell_possibilities(working_square[row_start + t.0][col_start + t.1]) > 0)
				).collect();
				
				for t in targets {
					let target = working_square[row_start + t.0][col_start + t.1];
					for k in 0..side_length {
						if 1 << k & target > 0 {
							row_possibility_array[k][t.0] = 1;
							col_possibility_array[k][t.1] = 1;
						}
					}
				}

				for k in 0..side_length {
					if row_possibility_array[k].iter().sum::<usize>() == 1 {
						let mut i = 0;
						while row_possibility_array[k][i] == 0 { i += 1; }

						let value = u128::try_from(k).unwrap();

						for col in 0..col_start {
							working_square[i][col] &= !value;
						}

						for col in col_start + 3..side_length {
							working_square[i][col] &= !value;
						}
					}

					if col_possibility_array[k].iter().sum::<usize>() == 1 {
						let mut i = 0;
						while col_possibility_array[k][i] == 0 { i += 1; }

						let value = u128::try_from(k).unwrap();

						for row in 0..col_start {
							working_square[row][i] &= !value;
						}

						for row in col_start + 3..side_length {
							working_square[row][i] &= !value;
						}
					}
				}
			}
			
			return working_square;
		}

		fn heuristic_place_and_update(&self, working_square: &mut Vec<Vec<UBitRep>>, value: &UBitRep, i: usize, j: usize) -> bool {
			//for now assume 9
			let side_length = working_square.len();
			let mut cell_row = i / 3;
			cell_row *= 3;
			let mut cell_col = j / 3;
			cell_col *= 3;

			let mut coords: Vec<(usize, usize)> = Vec::with_capacity(side_length - 1);

			for row in cell_row..cell_row + 3 {
				for col in cell_col..cell_col + 3 {
					if row == i && col == j { continue; }
					coords.push((row, col));
				}
			}

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
						self.place_and_update(working_square, &new_value, coord.0, coord.1);
					}	
				}
			}
			true
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
	pub fn cull<T>(mut working_square: Vec<Vec<T>>, n: usize) -> Vec<Vec<T>> where T: std::marker::Copy + std::cmp::PartialEq + std::ops::Sub<Output = T> {
		// let mut working_square = square.clone();
		let side_length = working_square.len();
		let zero: T = working_square[0][0] - working_square[0][0];

		//This is probably quicker than infinite misses on generating random coordinates.
		let mut valid_matrix_indices: Vec2D<usize> = (0..side_length).map(
			|i| (0..side_length).filter(
				|&j| working_square[i][j] != zero
			).collect()
		).collect();
		
		let mut valid_row_indices: Vec<usize> = (0..side_length).filter(
			|&i| valid_matrix_indices[i].len() != 0
		).collect();
		
		let mut k = 0;
		while valid_row_indices.len() != 0 && k < n {
			let relative_i = rand_usize(k) % valid_row_indices.len();
			let i = valid_row_indices[relative_i];
			let relative_j = rand_usize(k) % valid_matrix_indices[i].len();
			let j = valid_matrix_indices[i][relative_j];

			working_square[i][j] = zero;

			valid_matrix_indices[i].remove(relative_j);
			if valid_matrix_indices[i].len() == 0 {
				valid_row_indices.remove(relative_i);
			}

			k += 1;
		}

		return working_square;
	}
	
	pub fn to_bit(val: UDisplayRep) -> UBitRep {
		let output = UBitRep::from(val);
		if output != 0 {
			return 1 << (output - 1);
		}
		output //0
	}

	pub fn bit_format(v: Vec2D<UDisplayRep>) -> Vec2D<UBitRep> {
		//Turn a non-zero x into 2 ^^ (x - 1), else 0.
		v.iter().map(|line|
			line.iter().map(|x|
				to_bit(*x)
			).collect()
		).collect()
	}

	pub fn to_display(val: UBitRep) -> UDisplayRep {
		let mut output = 0;
		if count_cell_possibilities(val) > 1 { return 0; }
		if val == 0 { return 0; }

		loop {
			if val & 1 << output > 0 { return output + 1; }
			output += 1;
		}
	}

	pub fn display_format(v: Vec<Vec<UBitRep>>) -> Vec<Vec<UDisplayRep>> {
		v.iter().map(|line|
			line.iter().map(|x|
				to_display(*x)
			).collect()
		).collect()
	}
	
	pub fn count_cell_possibilities(mut target: UBitRep) -> UBitRep {
		// let mut target = val;
		let mut count = 0;
		
		//Kernighan's Algorithm
		while target != 0 {
			target = target & (target - 1);
			count += 1;
		}
		count
	}

	pub fn gen_square(side_length: usize) -> Vec2D<UDisplayRep> {
		let n = UDisplayRep::try_from(side_length).unwrap();
		let mut square: Vec2D<UDisplayRep> = Vec::with_capacity(side_length);

		for i in 1..UDisplayRep::try_from(1 + side_length).unwrap() {
			let new_line = (i..n + 1).chain(1..i).collect();
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

	pub fn gen_sudoku(side_length: usize) -> Vec2D<UDisplayRep> {
		let n = UDisplayRep::try_from(side_length).unwrap();
		if n != 9 { return Vec::new(); }
		
		let prototype_line: Vec<UBitRep> = (0..n).into_iter().map(|x| 1 << x).collect();
		let unconstrained_value = prototype_line.iter().sum();
		let mut possibilities: Vec<UBitRep> = (0..n).into_iter().map(|_x| unconstrained_value).collect();
		let mut square: Vec<Vec<UBitRep>> = Vec::with_capacity(side_length);
		let mut sudoku_possibilities = possibilities.clone();
		
		for i in 0..(side_length - 1) {
			let mut new_line = prototype_line.clone();

			let mut count = 0;
			loop {
				if generate_valid_line(&mut new_line, &sudoku_possibilities) { break; }
				count += 1;
				if count == 100 { return Vec::new(); }
			}
			
			square.push(new_line);

			for j in 0..side_length {
				possibilities[j] &= !square[i][j];
				sudoku_possibilities[j] &= !square[i][j];
			}

			if i % 3 == 2 {
				sudoku_possibilities = possibilities.clone();
			} else {
				for j in 0..3 {
					let line = &square[i];
					let new_constraints = line[j * 3] | line[j * 3 + 1] | line[j * 3 + 2];

					sudoku_possibilities[j * 3    ] &= !new_constraints;
					sudoku_possibilities[j * 3 + 1] &= !new_constraints;
					sudoku_possibilities[j * 3 + 2] &= !new_constraints;
				}
			}
		}

		square.push(possibilities);
		display_format(square)
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
			
			while misses + i < n {
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

#[cfg(test)]
mod tests {
	use moon_stats::moon_stats::{ StandardlyDistributable, StandardDistribution };
	use super::latin_square::*;
	use std::time::Instant;

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
			let latin_square = gen_square(n);
			super::tui::print_2d_vec(&latin_square);
			validate_latin_square(&latin_square);
		}
	}

	#[test]
	#[ignore]
	fn generation_distribution_statistics() {
		let samples = 200;
		let size = 60;
		let vector_store: Vec<Vec<Vec<UDisplayRep>>> = (0..samples).map(|_x| gen_square(size)).collect();

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
			
			let _results: Vec<Vec<Vec<UDisplayRep>>> = (0..samples).into_iter().map(|_x| gen_square(n)).collect();

			cum_time_millis = total_time.elapsed().as_millis() - cum_time_millis;
			println!("N: {:>3} | {:>5} |", n, cum_time_millis);
		}

		println!("Test suite ran for {} seconds.", total_time.elapsed().as_secs());
    }
	
	#[test]
	#[should_panic]
	fn oversize_square() {
		gen_square(usize::MAX);
	}

	#[test]
	fn test_latin_culler() {
		let n = 6;
		let mut avgs: Vec<UDisplayRep> = (0..(n * n)).into_iter().map(|_x| 0).collect();
		//Magic number of samples at which point the average coalesces.;
		let samples = 5000;
		let cull_quantity = n * n / 2;

		super::tui::print_2d_vec(&cull(gen_square(n), cull_quantity));
		
		for _i in 0..samples {

			let latin_square = bit_format(gen_square(n));
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
		let mut latin_square = bit_format(gen_square(5));
		let test_vec = latin_square[0].clone();

		for i in 1..latin_square.len() {
			latin_square[0][i] |= latin_square[0][0];
		}

		let i = 1;
		let j = 0;
		latin_square[0][0] |= latin_square[i][j];
		let val = latin_square[i][j];
		latin_square[i][j] = 31;
		
		let latin = DefaultLatinSquare {
			square: display_format(latin_square.clone()),
		};


		super::tui::print_2d_vec(&latin.square);
		latin.place_and_update(&mut latin_square, &val, i, j);
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

		let latin = DefaultLatinSquare {
			square: latin_square
		};

		let latin_squares = latin.solve();
		if latin_squares.len() > 0 {
			super::tui::print_2d_vec(&latin_squares[0]);
		}
		println!("\nTEST_LATIN_SOLVER: {}", latin_squares.len());
	}

	#[test]
	fn test_sudoku_solver() {
		let n = 9;
		let mut latin_square = gen_sudoku(n);

		println!("Initial square...");
		super::tui::print_2d_vec(&latin_square);
		println!();

		latin_square = cull(latin_square, 2 * n * n / 5);
		super::tui::print_2d_vec(&latin_square);
		println!();
		// let latin_squares = solve(latin_square);

		let latin = Sudoku {
			square: latin_square
		};

		let latin_squares = latin.solve();
		if latin_squares.len() > 0 {
			super::tui::print_2d_vec(&latin_squares[0]);
		}
		println!("\nTEST_LATIN_SOLVER: {}", latin_squares.len());
	}
}
