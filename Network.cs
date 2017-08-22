using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NN{
    public class NeuronNetwork{
        int inputCount;
        int hiddenCount;
        int outputCount;
        double learningRate = 0.05;
        double momentum = 0.75;
        Matrix<double> inputWeights;
        Matrix<double> hiddenLayerWeights;

        public NeuronNetwork(int inputCount, int hiddenCount, int outputCount, double learningRate){
            this.inputCount = inputCount;
            this.hiddenCount = hiddenCount;
            this.outputCount = outputCount;
            inputWeights = MatrixHelpers.InitialiseWeights(inputCount, hiddenCount, 2);
            hiddenLayerWeights = MatrixHelpers.InitialiseWeights(hiddenCount, outputCount, 2);
        }

        public void BackPropagate(double[] input, double[] expectedOutput){
            //Forward propagate
            double[] hiddenLayerValues = new double[hiddenCount];
            for (int i = 0; i < inputCount; i++){
                for (int j = 0; j < hiddenCount; j++){
                    // + 1 is bias
                    hiddenLayerValues[j] = inputWeights[i, j] * input[i] + 1;
                }
            }
            hiddenLayerValues = MatrixHelpers.HyperTanActivation(hiddenLayerValues);
            double[] output = new double[outputCount];
            for (int i = 0; i < hiddenCount; i++){
                for (int j = 0; j < outputCount; j++){
                    output[j] = hiddenLayerWeights[i, j] * hiddenLayerValues[i] + 1;
                }
            }
            output = MatrixHelpers.HyperTanActivation(output);
            //Calculate error
            double[] outputError = MatrixHelpers.CalculateOutputError(output, expectedOutput);
            //Backpropagate
            //Calculate hidden layer error
            double[] hiddenLayerError = new double[hiddenCount];
            for (int i = 0; i < hiddenCount; i++){
                for (int j = 0; j < outputCount; j++){
                    hiddenLayerError[i] += hiddenLayerWeights[i, j] * outputError[j];
                }
            }
            for (int i = 0; i < hiddenCount; i++){
                hiddenLayerError[i] *= (1-Math.Pow(Math.Tanh(hiddenLayerValues[i]), 2))/2;
            }

        }
    }
}