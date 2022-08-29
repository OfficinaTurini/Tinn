#pragma once

#include <stdio.h>
/*! Copyright 2017-2022 Officina Turini\n
This file is part of otNeuralNetwork project.\n

otStudio is free software: you can redistribute it and/or modify\n
it under the terms of the GNU General Public License as published by\n
the Free Software Foundation, either version 3 of the License.n
\n
otStudio is distributed in the hope that it will be useful,\n
but WITHOUT ANY WARRANTY; without even the implied warranty of\n
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n
GNU General Public License for more details.\n
\n
You should have received a copy of the GNU General Public License\n
along with otNeuralNetwork.  If not, see <http://www.gnu.org/licenses/>.
*/

#if defined(_WIN32) || defined(__WIN32__)
	#if defined(_MSC_VER)
		#ifdef OTNN_STATIC_LIBRARY
			#define OTNN_LIB
		#else
			#ifdef OTNN_SOURCE_CODE
				#define OTNN_LIB __declspec(dllexport)
			#else
				#define OTNN_LIB __declspec(dllimport)
			#endif
		#endif
	#else
	#if defined(__BORLANDC__)
		#ifdef OTNN_SOURCE_CODE
			#define OTNN_LIB _export
		#else
			#define OTNN_LIB _import
		#endif
	#endif
	#endif
#else
// UNIX

#endif

class OTNN_LIB otTinn
{
public:
	otTinn(int nips, int nops, int nhid);
	~otTinn();

	float			* predict(const float * in);	///< Returns an output prediction given an input.
	float			train(const float * in, const float * tg, float rate);	///< Trains a otTinn with an input and target output with a learning rate. Returns target to output error.
	bool			save(const char * path);		///< Saves a otTinn to disk.
	bool			load(const char * path);		///< Loads a otTinn from disk.
	void			print(const float * arr, const int size);	///< Prints an array of floats. Useful for printing predictions.

private:
	void			wbrand();	///< Randomizes otTinn weights and biases.
	float			frand();	///< Returns floating point random from 0.0 - 1.0.
	inline float	err(const float a, const float b);		///< Computes error.
	inline float	pderr(const float a, const float b);	///< Returns partial derivative of error function.
	float			toterr(const float * const tg, const float * const o, const int size);	///< Computes total error of target to output.
	float			act(const float a);						///< Activation function.
	float			pdact(const float a);					///< Returns partial derivative of activation function.
	void			bprop(const float * const in, const float * const tg, float rate);	///< Performs back propagation.
	void			fprop(const float * const in);			///< Performs forward propagation.

	float	* _w;	///< All the weights.
	float	* _x;	///< Hidden to output layer weights.
	float	* _b;	///< Biases.
	float	* _h;	///< Hidden layer.
	float	* _o;	///< Output layer.
	int		_nb;	///< Number of biases - always two - otTinn only supports a single hidden layer.
	int		_nw;	///< Number of weights.
	int		_nips;	///< Number of inputs.
	int		_nhid;	///< Number of hidden neurons.
	int		_nops;	///< Number of outputs.
};

class OTNN_LIB otNeuralFramework
{
public:
	otNeuralFramework(int nips, int nops, int nhid);
	~otNeuralFramework();

	bool	dataset(const char * path);
	void	shuffle();
	float	training(otTinn & tin, float rate);
	float	error() { return _error; }
	int		rows() { return _rows; }
	int		outputs() { return _nops; }
	const float	* input(int row = 0) { return _in[row]; }
	const float	* target(int row = 0) { return _tg[row]; }

private:
	float	** new2d(const int rows, const int cols);	///< Create new 2D array of floats.
	int		lines(FILE * const file);					///< Returns the number of lines in a file.
	char	* readln(FILE * const file);				///< Reads a line from a file.
	void	parse(char * line, const int row);			///< Gets one row of inputs and outputs from a string.
	void	dfree();									///< Frees a data object from the heap.

	int		_nips;	///< Number of inputs to neural network.
	int		_nops;	///< Number of outputs to neural network.
	int		_nhid;	///< Number of hidden layers
	int		_rows;	///< Rows in training file (number of sets for neural network).
	float	** _in;	///< 2D floating point array of input.
	float	** _tg;	///< 2D floating point array of target.
	float	_error;
};

