#include "otNeuralNetwork.h"

#define N_INPUTS	256
#define N_OUTPUTS	10
#define N_INLAYER	28
#define ITERATIONS	128

void main(int argc, char argv[])
{
	otNeuralFramework	nn(N_INPUTS, N_OUTPUTS, N_INLAYER);
	otTinn				tinn(N_INPUTS, N_OUTPUTS, N_INLAYER);

	if(nn.dataset("semeion.data"))
	{
		// The number of iterations can be changed for stronger training.
		float	rate = 1.0f, error;
		// Hyper Parameters.
		// Learning rate is annealed and thus not constant.
		// It can be fine tuned along with the number of hidden layers.
		// Feel free to modify the anneal rate.
		const float anneal = 0.99f;
		for(int i = 0; i < ITERATIONS; i++)
		{
			nn.shuffle();
			error = nn.training(tinn, rate);
			printf("[%5u] error %.12f :: learning rate %f\n", i, (double) error / nn.rows(), (double) rate);
			rate *= anneal;
		}
	}
	else
	{
		printf("Error: Can't find semeion.data file!\nGet it from: http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data\n");
		return;
	}
	// This is how you save the neural network to disk.
	if(tinn.save("saved.tinn"))
	{
		// This is how you load the neural network from disk.
		if(tinn.load("saved.tinn"))
		{
			// Now we do a prediction with the neural network we loaded from disk.
			// Ideally, we would also load a testing set to make the prediction with,
			// but for the sake of brevity here we just reuse the training set from earlier.
			// One data set is picked at random (zero index of input and target arrays is enough
			const float * pd = tinn.predict(nn.input());
			const float * tg = nn.target();
			
			// Prints target.
			tinn.print(tg, nn.outputs());

			// Prints prediction.
			tinn.print(pd, nn.outputs());

			printf("Test complete! Press enter to exit.\n");
			getchar();
		}
	}
}