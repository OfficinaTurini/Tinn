#include "otNeuralNetwork.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

otTinn::otTinn(int nips, int nops, int nhid)
{
	// otTinn only supports one hidden layer so there are two biases.
	_nb = 2;
	_nw = nhid * (nips + nops);
	_w = (float *) calloc(_nw, sizeof(*_w));
	_x = _w + nhid * nips;
	_b = (float *) calloc(_nb, sizeof(*_b));
	_h = (float *) calloc(nhid, sizeof(*_h));
	_o = (float*) calloc(nops, sizeof(*_o));
	_nips = nips;
	_nhid = nhid;
	_nops = nops;
	wbrand();
}

otTinn::~otTinn()
{
	free(_w);
	free(_b);
	free(_h);
	free(_o);
}

void otTinn::wbrand()
{
	for(int i = 0; i < _nw; i++) _w[i] = frand() - 0.5f;
	for(int i = 0; i < _nb; i++) _b[i] = frand() - 0.5f;
}

float otTinn::frand()
{
	return rand() / (float) RAND_MAX;
}

float otTinn::pderr(const float a, const float b)
{
	return a - b;
}

float otTinn::toterr(const float * const tg, const float * const o, const int size)
{
	float sum = 0.0f;
	for(int i = 0; i < size; i++)
		sum += err(tg[i], o[i]);
	return sum;
}

float otTinn::err(const float a, const float b)
{
	return 0.5f * (a - b) * (a - b);
}

float otTinn::act(const float a)
{
	return 1.0f / (1.0f + expf(-a));
}

float otTinn::pdact(const float a)
{
	return a * (1.0f - a);
}

void otTinn::bprop(const float * const in, const float * const tg, float rate)
{
	for(int i = 0; i < _nhid; i++)
	{
		float sum = 0.0f;
		// Calculate total error change with respect to output.
		for(int j = 0; j < _nops; j++)
		{
			const float a = pderr(_o[j], tg[j]);
			const float b = pdact(_o[j]);
			sum += a * b * _x[j * _nhid + i];
			// Correct weights in hidden to output layer.
			_x[j * _nhid + i] -= rate * a * b * _h[i];
		}
		// Correct weights in input to hidden layer.
		for(int j = 0; j < _nips; j++)
			_w[i * _nips + j] -= rate * sum * pdact(_h[i]) * in[j];
	}
}

void otTinn::fprop(const float * const in)
{
	// Calculate hidden layer neuron values.
	for(int i = 0; i < _nhid; i++)
	{
		float	sum = 0.0f;
		for(int j = 0; j < _nips; j++)
			sum += in[j] * _w[i * _nips + j];
		_h[i] = act(sum + _b[0]);
	}
	// Calculate output layer neuron values.
	for(int i = 0; i < _nops; i++)
	{
		float	sum = 0.0f;
		for(int j = 0; j < _nhid; j++)
			sum += _h[j] * _x[i * _nhid + j];
		_o[i] = act(sum + _b[1]);
	}
}

float * otTinn::predict(const float * in)
{
	fprop(in);
	return _o;
}

float otTinn::train(const float * in, const float * tg, float rate)
{
	fprop(in);
	bprop(in, tg, rate);
	return toterr(tg, _o, _nops);
}

bool otTinn::save(const char * path)
{
	FILE	* const file = fopen(path, "w");
	if(file == NULL)
		return false;
	// Save header.
	fprintf(file, "%d %d %d\n", _nips, _nhid, _nops);
	// Save biases and weights.
	for(int i = 0; i < _nb; i++) fprintf(file, "%f\n", (double) _b[i]);
	for(int i = 0; i < _nw; i++) fprintf(file, "%f\n", (double) _w[i]);
	fclose(file);
	return true;
}

bool otTinn::load(const char * const path)
{
	FILE	* const file = fopen(path, "r");
	if(file == NULL)
		return false;
	_nips = 0;
	_nhid = 0;
	_nops = 0;
	// Load header.
	fscanf(file, "%d %d %d\n", &_nips, &_nhid, &_nops);
	
	free(_w);
	free(_b);
	free(_h);
	free(_o);

	// Build a new otTinn.
	_nw = _nhid * (_nips + _nops);
	_w = (float *) calloc(_nw, sizeof(*_w));
	_x = _w + _nhid * _nips;
	_b = (float *) calloc(_nb, sizeof(*_b));
	_h = (float *) calloc(_nhid, sizeof(*_h));
	_o = (float *) calloc(_nops, sizeof(*_o));
	// Load bias and weights.
	for(int i = 0; i < _nb; i++) fscanf(file, "%f\n", &_b[i]);
	for(int i = 0; i < _nw; i++) fscanf(file, "%f\n", &_w[i]);
	fclose(file);
	return true;
}

void otTinn::print(const float * arr, const int size)
{
	for(int i = 0; i < size; i++)
		printf("%f ", (double) arr[i]);
	printf("\n");
}

otNeuralFramework::otNeuralFramework(int nips, int nops, int nhid)
{
	_nips = nips;
	_nops = nops;
	_nhid = nhid;
	_in = 0;
	_tg = 0;
	_rows = -1;
	_error = 0.0f;
}

otNeuralFramework::~otNeuralFramework()
{
}

float ** otNeuralFramework::new2d(const int rows, const int cols)
{
	float	** row = (float **) malloc((rows) * sizeof(float *));
	for(int r = 0; r < rows; r++)
		row[r] = (float *) malloc((cols) * sizeof(float));
	return row;
}

int otNeuralFramework::lines(FILE * const file)
{
	int	ch = EOF;
	int lines = 0;
	int pc = '\n';
	while((ch = getc(file)) != EOF)
	{
		if(ch == '\n')
			lines++;
		pc = ch;
	}
	if(pc != '\n')
		lines++;
	rewind(file);
	return lines;
}

char * otNeuralFramework::readln(FILE * const file)
{
	int		ch = EOF;
	int		reads = 0;
	int		size = 1024;
	char	* line = (char *) malloc((size) * sizeof(char));
	while((ch = getc(file)) != '\n' && ch != EOF)
	{
		line[reads++] = ch;
		if(reads + 1 == size)
			line = (char *) realloc((line), (size *= 2) * sizeof(char));
	}
	line[reads] = '\0';
	return line;
}

void otNeuralFramework::parse(char * line, const int row)
{
	const int cols = _nips + _nops;
	for(int col = 0; col < cols; col++)
	{
		const float val = (float) atof(strtok(col == 0 ? line : NULL, " "));
		if(col < _nips)
			_in[row][col] = val;
		else
			_tg[row][col - _nips] = val;
	}
}

bool otNeuralFramework::dataset(const char * path)
{
	bool	ret = false;
	FILE	* file = fopen(path, "r");
	if(file == NULL)
		return ret;
	_rows = lines(file);

	if(_rows >= 1)
	{
		_in = new2d(_rows, _nips);
		_tg = new2d(_rows, _nops);

		if(_in && _tg)
		{
			for(int row = 0; row < _rows; row++)
			{
				char	* line = readln(file);
				parse(line, row);
				free(line);
			}
			fclose(file);
			ret = true;
		}
	}
	return ret;
}

void otNeuralFramework::dfree()
{
	for(int row = 0; row < _rows; row++)
	{
		free(_in[row]);
		free(_tg[row]);
	}
	free(_in);
	free(_tg);
}

void otNeuralFramework::shuffle()
{
	for(int a = 0; a < _rows; a++)
	{
		const int b = rand() % _rows;
		float	* ot = _tg[a];
		float	* it = _in[a];
		// Swap output.
		_tg[a] = _tg[b];
		_tg[b] = ot;
		// Swap input.
		_in[a] = _in[b];
		_in[b] = it;
	}
}

float otNeuralFramework::training(otTinn & tin, float rate)
{
	_error = 0;
	for(int j = 0; j < _rows; j++)
		_error += tin.train(_in[j], _tg[j], rate);
	return _error;
}