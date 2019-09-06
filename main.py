from numpy import *
import pandas as pd

def compute_error_from_points(b, m, points):
	#error is initialy 0
	totalError = 0
	#for loop, iterate every point
	for i in range(0, len(points)):
		#get x and y value
		x = points[i, 0]
		y = points[i, 1]
 
		#calculate the difference and add it
		totalError += (y - (m*x-b))**2

	return totalError/float(len(points))


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m

	#gradient descent
	for i in range(num_iterations):
		#update m and b with the new parameters from the gradient descent step
		b, m = step_gradient(b, m, array(points), learning_rate)

	return [b,m]

def step_gradient(current_b, current_m, points, learning_rate):
	b_gradient=0
	m_gradient=0
	N = float(len(points))

	for i in range(0, len(points)):
		x = points[i,0]
		y = points[i,1]
		b_gradient += -(2/N)*(y-((current_m*x)+current_b))
		m_gradient += -(2/N)*x*(y-((current_m*x)+current_b))

	new_b = current_b - (learning_rate*b_gradient)
	new_m = current_m - (learning_rate*m_gradient)
	return [new_b, new_m]


def run():
	#step 1 -- collect our data
	points = genfromtxt('data.csv', delimiter=',')

	#print(points)

	#get the good colums with the rigth data in it


	#step 2 -- define hyper parameters
	learning_rate = 0.0001 #how fast will it learn
	initial_b = 0
	initial_m = 0
	num_iterations = 1000

	#step 3 -- train our model
	print("Start gradient descent at b={0}, m={1} and error={2}".format(initial_b, initial_m,compute_error_from_points(initial_b, initial_m, points)))
	[b,m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate,num_iterations)
	print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_from_points(b, m, points)))

if __name__ == '__main__':
	run()