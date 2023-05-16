using System;
using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using System.Web;


namespace SimpleNeuralNetwork
{
    internal class Program
    {
        static public double[] TrainingData = { 0.66, 0.57, 0.46, 0.54, 0.59, 0.73, 0.76, 0.58, 0.26, 0.38, 0.8, 0.41, 0.28, 0.91, 0.62, 0.26, 0.67, 0.11, 0.08, 0.67, 0.98, 0.76, 0.22, 0.61, 0.91, 0.87, 0.17, 0.9, 0.84, 0.08, 0.3, 0.29, 0.51, 0.14, 0.73, 0.54, 0.9, 0.15, 0.79, 0.28, 0.36, 0.49, 0.41, 0.48, 0.76, 0.8, 0.15, 0.96, 0.18, 0.12, 0.53, 0.35, 0.63, 0.18, 0.41, 0.58, 0.13, 0.53, 0.49, 0.17, 0.11, 0.62, 0.63, 0.2, 0.94, 0.55, 0.77, 0.96, 0.05, 0.79, 0.6, 0.11, 0.15, 0.08, 0.52, 0.3, 0.14, 0.01, 0.35, 0.85, 0.98, 0.17, 0.63, 0.97, 0.96, 0.41, 0.64, 0.48, 0.59, 0.79, 0.04, 0.13, 0.17, 0.47, 0.54, 0.33, 0.09, 0.41, 0.1, 0.79, 0.53, 0.22, 0.72, 0.41, 0.13, 0.53, 0.49, 0.75, 0.52, 0.45, 0.46, 0.39, 0.68, 0.77, 0.83, 0.68, 0.25, 0.03, 0.64, 0.09, 0.77, 0.25, 0.23, 0.79, 0.25, 0.26, 0.97, 0.31, 0.78, 0.92, 0.17, 0.57, 0.81, 0.04, 0.02, 0.08, 0.61, 0.06, 0.3, 0.63, 0.98, 0.72, 0.81, 0.41, 0.63, 0.92, 0.28, 0.79, 0.89, 0.32, 0.06, 0.9, 0.89, 0.92, 0.31, 0.61, 0.05, 0.5, 0.08, 0.36, 0.03, 0.77, 0.36, 0.25, 0.04, 0.97, 0.58, 0.85, 0.1};

        public static double SampleGaussian(Random random, double mean, double stddev)
        {
            // The method requires sampling from a uniform random of (0,1]
            // but Random.NextDouble() returns a sample of [0,1).
            double x1 = 1 - random.NextDouble();
            double x2 = 1 - random.NextDouble();

            double y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
            return y1 * stddev + mean;
            
        }

        static List<double> OutputData = new List<double>();


        static void Main(string[] args)
        {
            Program program = new Program();
            program.DataProcess();
            

            List<neurone> hiddenLayer = new List<neurone>();

            hiddenLayer.Capacity = 5;

            Random random= new Random();

            int temp = 0;
            int Ccount = 0;


            double tempData = 0;
            List<double> Weights = new List<double>();
            List<double> OutputWeights = new List<double>();
            List<double> TempOutputData = new List<double>();
            List<double> Data = new List<double>();
            List<double> TempWeights = new List<double>();

            string WeightFile = File.ReadAllText("Weights.txt");

            string[] str = WeightFile.Split();


            for (int i = 0; i < 20; i++)
            {
                Weights.Add(Convert.ToDouble(str[i]));

            }



            foreach (double D in TrainingData)
            {
                Data.Add(D);
                temp++;


                if (temp == 4)
                {
                    for (int j = 1; j < 5; j++) 
                    {
                        for(int i = 1; i <= 4; i++)
                        {
                            TempWeights.Add(Weights[j * i]);
                        }
                        hiddenLayer.Add(new neurone(Data, TempWeights));
                        TempWeights.Clear();
                    }

                    temp = 0;
                    Data.Clear();
                }
                

                for (int i = 0; i < hiddenLayer.Count; i++)
                {
                    neurone N = hiddenLayer[i];
                    N.processor();

                    OutputData.Add(N.Output());

                }

                for(int i = 0; i < TempOutputData.Count; i++)
                {
                    OutputWeights.Add(SampleGaussian(random, 0, 1));

                    tempData += (TempOutputData[i] * OutputWeights[i]);
                    
                }
                OutputData.Add(tempData);
                Ccount++;
                hiddenLayer.Clear();
            }



            dataUnprocess();
            Console.WriteLine(string.Join(",", OutputData));
            Console.WriteLine(OutputData.Count);
            Console.WriteLine(TrainingData.Count());


            
        }

        public static double ReLU(double x) //Activation function, beings non linearality into the network
        {
            return Math.Max(0, x);
            
        }

        static void dataUnprocess()
        {

            for(int i = 0; i < OutputData.Count; i++)
            {
                OutputData[i] = OutputData[i] * 100;
            }
        }



        void DataProcess() // makes the sums for the matracies
        {
            List<int> Sums = new List<int>();

            int temp = 0;
            int temp2 = 0;

            foreach(double i in TrainingData)
            {
                if(temp2 == 4)
                {
                    temp2 = 0;
                    Sums.Add(temp);
                    temp = 0;
                }
                temp2++;
                temp += Convert.ToInt32(i * 100);
            }

            if (temp > 0)
            {
                Console.WriteLine(temp2);
            }

            Console.WriteLine(String.Join("," ,Sums) + ": is the sums of the matracies");
            Console.WriteLine(Sums.Count);
        }


    }


    public class neurone // neurone class, defines the properties of the neurones
    {
        Program program = new Program();

        List<double> input = new List<double>();
        List<double> weight = new List<double>();
        double output
        { get; set; }

        double bias
        { get; set; }

            
        public neurone(List<double> Input , List<double> Weight )
        {
            input = Input.ToList();
            weight = Weight.ToList();
            bias = 0;
        }
   
        public void processor() // processes the input into the output data
        { 

            for(int i = 0; i < input.Count; i++)
            {
                output += (input[i] * weight[i]);
            }

            output += bias;

            output =  Program.ReLU(output);

        }

        public double Output()
        {
            return output;
        }

        
    }

}
