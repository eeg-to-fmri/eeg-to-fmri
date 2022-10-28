import numpy as np
import z3
import math


class conv_sat:

	def __init__(self, input_shape, output_shape, next_input_shape=None):

		self.conv_solver = z3.Solver()

		self.kernel_1 = z3.Int('k_1')
		self.stride_1 = z3.Int('s_1')

		self.is_tuple = False

		if(type(input_shape) is tuple):
			self.is_tuple = True

			

			self.kernel_2 = z3.Int('k_2')
			self.stride_2 = z3.Int('s_2')

			self.out_1 = z3.Int('out_1')
			self.out_2 = z3.Int('out_2')
			
			self.input_shape_1 = int(input_shape[0])
			self.input_shape_2 = int(input_shape[1])

			self.output_shape = int(output_shape)

			if(self.input_shape_1*self.input_shape_2 < self.output_shape):
				self.gt=True
				if(next_input_shape == None):
					next_input_shape = (140000, 140000)
			else:
				self.gt=False
				if(next_input_shape == None):
					next_input_shape = (1, 1)

		else:
			
			
			if(input_shape < output_shape):
				self.input_shape = int(input_shape)
				self.output_shape = int(output_shape)
				if(next_input_shape == None):
					next_input_shape = (140000, 140000)
			else:
				self.input_shape = int(output_shape)
				self.output_shape = int(input_shape)
				if(next_input_shape == None):
					next_input_shape = (1, 1)

		self.next_input_shape_1 = next_input_shape[0]
		self.next_input_shape_2 = next_input_shape[1]

		self.possible_combinations = []

	def setup_restrictions(self):

		if(self.is_tuple):
			self.conv_solver.add(self.kernel_1 - self.stride_1 >= 0)
			self.conv_solver.add(self.kernel_2 - self.stride_2 >= 0)
			self.conv_solver.add(self.kernel_1 > 0)
			self.conv_solver.add(self.stride_1 > 0)
			self.conv_solver.add(self.kernel_2 > 0)
			self.conv_solver.add(self.stride_2 > 0)			

			if(self.gt):
				self.conv_solver.add( ( (self.input_shape_1 - 1)*self.stride_1 + self.kernel_1 ) == self.out_1)
				self.conv_solver.add( ( (self.input_shape_2 - 1)*self.stride_2 + self.kernel_2 ) == self.out_2)

				self.conv_solver.add(self.out_1 * self.out_2 == self.output_shape)

				self.conv_solver.add( self.out_1 <= self.next_input_shape_1)
				self.conv_solver.add( self.out_2 <= self.next_input_shape_2)
			else:
				self.conv_solver.add( ( (self.out_1 - 1)*self.stride_1 + self.kernel_1 ) == self.input_shape_1)
				self.conv_solver.add( ( (self.out_2 - 1)*self.stride_2 + self.kernel_2 ) == self.input_shape_2)

				self.conv_solver.add(self.out_1 * self.out_2 == self.output_shape)

				self.conv_solver.add( self.out_1 >= self.next_input_shape_1)
				self.conv_solver.add( self.out_2 >= self.next_input_shape_2)
		else:

			self.conv_solver.add(self.kernel_1 - self.stride_1 >= 0)
			self.conv_solver.add(self.kernel_1 > 0)
			self.conv_solver.add(self.stride_1 > 0)
			self.conv_solver.add((self.input_shape - 1)*self.stride_1 + self.kernel_1 == self.output_shape)


	def add_solution(self):
		if(self.is_tuple):
			kernel_1 = self.conv_solver.model()[self.kernel_1].as_long()
			kernel_2 = self.conv_solver.model()[self.kernel_2].as_long()
			stride_1 = self.conv_solver.model()[self.stride_1].as_long()
			stride_2 = self.conv_solver.model()[self.stride_2].as_long()

			self.possible_combinations += [((kernel_1, kernel_2), (stride_1, stride_2))]
			
			self.conv_solver.add(z3.Or(self.kernel_2 != kernel_2, 
				z3.Or(self.kernel_2 != kernel_2,
					z3.Or(self.kernel_1 != kernel_1, 
						self.stride_1 != stride_1))))


		else:
			kernel_1 = self.conv_solver.model()[self.kernel_1].as_long()
			stride_1 = self.conv_solver.model()[self.stride_1].as_long()

			self.possible_combinations += [(kernel_1, stride_1)]
			
			self.conv_solver.add(z3.Or(self.kernel_1 != kernel_1, self.stride_1 != stride_1))



	def solve(self):
		self.setup_restrictions()

		self.conv_solver.set("timeout", 60000)

		#while there are solutions to be returned
		while(self.conv_solver.check() == z3.sat):
			self.add_solution()

		return self.possible_combinations



if __name__ == "__main__":


	solver = conv_sat(20, 20)
	print(solver.solve())