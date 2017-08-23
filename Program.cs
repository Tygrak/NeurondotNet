using System;
using System.IO;
using System.Drawing;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;


namespace NN{
    class Program{
        static void Main(string[] args){
            Matrix<double> trainingFeatures;
            Matrix<double> trainingOutput;
            MatrixHelpers.ReadDataFile(@"iris_data_files/iris_training.dat", 4, 3, out trainingFeatures, out trainingOutput);
            NeuronNetwork nn = new NeuronNetwork(new int[]{4, 3, 3, 3});
            nn.Train(trainingFeatures, trainingOutput, 10000, 2.5);
            nn.PredictOutput(new double[]{0.813331, 0.692682, 0.854032, 0.975}, new double[]{0, 0, 1});
            double[] output = nn.GetOutput();
            Console.WriteLine("The networks output for input 0.813331, 0.692682, 0.854032, 0.975 is: " + MatrixHelpers.OutputToString(output));
            /*double[] testPredict = nn.PredictOutput(new double[]{0.813331, 0.692682, 0.854032, 0.975});
            Console.WriteLine("Networks prediction for values (0.813331, 0.692682, 0.854032, 0.975):");
            for (int i = 0; i < testPredict.Length; i++){
                Console.WriteLine(testPredict[i]);
            }*/
            Matrix<double> testFeatures;
            Matrix<double> testOutput;
            MatrixHelpers.ReadDataFile(@"iris_data_files/iris_test.dat", 4, 3, out testFeatures, out testOutput);
            Console.WriteLine("The networks error for the test set is: " + nn.CalculateSetError(testFeatures, testOutput).ToString("0.0000"));
        }
    }

    public class MatrixHelpers{
        public static string OutputToString(double[] output){
            string s = output[0].ToString("0.0000");
            for (int i = 1; i < output.Length; i++){
                s += ", " + output[i].ToString("0.0000");
            }
            return s;
        }

        public static int[] OutputToClass(Matrix<double> output){
            int[] classArr = new int[output.RowCount];
            for(int i = 0; i < output.RowCount; i++){
                for(int j = 0; j < output.ColumnCount; j++){
                    if(output[i, j] != 0){
                        classArr[i] = j+1;
                        break;
                    }
                }
            }
            return classArr;
        }
        public static Matrix<double> ClassToOutput(int[] classArr, int outputCount){
            Matrix<double> a = DenseMatrix.Create(classArr.Length, outputCount, 0);
            for(int i = 0; i < classArr.Length; i++){
                a[i, classArr[i]-1] = 1;
            }
            return a;
        }

        public static void ReadDataFile(string filename, int featuresCount, int outputCount, out Matrix<double> features, out Matrix<double> output){
            string[] lines = File.ReadAllLines(filename);
            features = DenseMatrix.Create(lines.Length, featuresCount, 0);
            output = DenseMatrix.Create(lines.Length, outputCount, 0);
            for(int i = 0; i < lines.Length; i++){
                string[] values = lines[i].Split(new char[0], StringSplitOptions.RemoveEmptyEntries);
                for(int j = 0; j < values.Length; j++){
                    if(j < featuresCount){
                        features[i, j] = Convert.ToDouble(values[j]);
                    } else{
                        output[i, j-featuresCount] = Convert.ToDouble(values[j]);
                    }
                }
            }
        }

        public static double HyperTanActivation(double input){
            return (Math.Tanh(input)+1)/2;
        }
        public static double[] HyperTanActivation(double[] inMatrix){
            for(int i = 0; i < inMatrix.Length; i++){
                inMatrix[i] = (Math.Tanh(inMatrix[i])+1)/2;
            }
            return inMatrix;
        }
        public static Matrix<double> HyperTanActivation(Matrix<double> inMatrix){
            for(int i = 0; i < inMatrix.RowCount; i++){
                for(int j = 0; j < inMatrix.ColumnCount; j++){
                    inMatrix[i, j] = (Math.Tanh(inMatrix[i, j])+1)/2;
                }
            }
            return inMatrix;
        }
        public static double HyperTanActivationDer(double input){
            return (1-Math.Pow(Math.Tanh(input), 2))/2;
        }
        public static double[] HyperTanActivationDer(double[] inMatrix){
            for(int i = 0; i < inMatrix.Length; i++){
                inMatrix[i] = (1-Math.Pow(Math.Tanh(inMatrix[i]), 2))/2;
            }
            return inMatrix;
        }
        public static Matrix<double> HyperTanActivationDer(Matrix<double> inMatrix){
            for(int i = 0; i < inMatrix.RowCount; i++){
                for(int j = 0; j < inMatrix.ColumnCount; j++){
                    inMatrix[i, j] = (1-Math.Pow(Math.Tanh(inMatrix[i, j]), 2))/2;
                }
            }
            return inMatrix;
        }

        public static Matrix<double> FeedForward(Matrix<double> inMatrix, Matrix<double> weightsMatrix, double bias){
            Matrix<double> output = Matrix.op_DotMultiply(weightsMatrix, inMatrix.Add(bias));
            return HyperTanActivation(output);
        }

        public static Matrix<double> InitialiseWeights(int width, int height, double maxWeight){
            Random r = new Random();
            Matrix<double> output = DenseMatrix.Create(width, height, 0);
            for(int i = 0; i < output.RowCount; i++){
                for(int j = 0; j < output.ColumnCount; j++){
                    output[i, j] = r.NextDouble() * (maxWeight - -maxWeight) + -maxWeight;
                }
            }
            return output;
        }

        public static double CalculateError(double[] output, double[] teacherOutput){
            double error = 0;
            for (int i = 0; i < output.Length; i++){
                error += Math.Pow(output[i]-teacherOutput[i], 2);
            }
            return Math.Sqrt(error)*0.5;
        }

        public static double[] CalculateOutputError(double[] output, double[] teacherOutput){
            double[] error = new double[output.Length];
            for (int i = 0; i < output.Length; i++){
                //error[i] = Math.Abs(output[i]-teacherOutput[i]);
                error[i] = output[i]-teacherOutput[i];
            }
            return error;
        }
    }
}
