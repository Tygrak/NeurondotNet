using System;
using System.IO;
using System.Drawing;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;


namespace NN{
    class Program{
        static void Main(string[] args){
            /*Matrix<double> a = DenseMatrix.Create(4, 3, 0);
            a[0, 0] = 1;
            a[1, 2] = 1;
            a[2, 0] = 1;
            a[3, 1] = 1;
            Console.WriteLine(a);
            int[] classa = MatrixHelpers.OutputToClass(a);
            for(int i = 0; i < classa.Length; i++){
                Console.WriteLine(classa[i]);
            }
            var b = MatrixHelpers.ClassToOutput(classa, 3);
            Console.WriteLine(b);*/
            Matrix<double> testFeatures;
            Matrix<double> testOutput;
            MatrixHelpers.ReadDataFile(@"iris_data_files/iris_test.dat", 4, 3, out testFeatures, out testOutput);
            NeuronNetwork nn = new NeuronNetwork(4, 2, 3, 0.05);
            double[] input = new double[]{0.282131, 0.269205, 0.430645, 0.427485}; 
            double[] output = nn.CalculateOutput(input);
            double[] actualOutput = new double[]{0, 1, 0};
            double[] outputError = MatrixHelpers.CalculateOutputError(output, actualOutput);
            Console.WriteLine("Output: " + output[0].ToString() + ", " + output[1].ToString() + ", " + output[2].ToString());
            Console.WriteLine("Actual output: " + actualOutput[0].ToString() + ", " + actualOutput[1].ToString() + ", " + actualOutput[2].ToString());
            Console.WriteLine("Error: " + MatrixHelpers.CalculateError(output, actualOutput).ToString());
            Console.WriteLine("Output error: " + outputError[0].ToString() + ", " + outputError[1].ToString() + ", " + outputError[2].ToString());
            //Console.WriteLine(testFeatures);
            //Console.WriteLine(testOutput);
        }
    }

    public class MatrixHelpers{
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
