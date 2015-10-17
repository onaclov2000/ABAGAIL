package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;
import func.nn.activation.*;
import shared.DataSet;
import shared.DataSetDescription;
import shared.reader.CSVDataSetReader;
import shared.reader.DataSetReader;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class AbaloneTest {
    private static Instance[] trainInstances = initializeInstances("Curious_George_train_features_100_percent.csv");
    private static Instance[] testInstances = initializeInstances("Curious_George_test_features.csv");

    private static int inputLayer = 12, hiddenLayer = 6, outputLayer = 1;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(trainInstances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        int it = args.length > 0 ? Integer.parseInt(args[0]): 1000;

        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer},
                new LogisticSigmoid());
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {

            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            for(int j = 0; j < it; j++) {
                oa[i].train();
            }
            end = System.nanoTime();

            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < trainInstances.length; j++) {
                networks[i].setInputValues(trainInstances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(trainInstances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;

            results +=  oaNames[i] + "," + it + "," + df.format(incorrect / (correct + incorrect)) + ","; 

            correct = 0;
            incorrect = 0;
            start = System.nanoTime();
            for(int j = 0; j < testInstances.length; j++) {
                networks[i].setInputValues(testInstances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(testInstances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime += end - start;
            testingTime /= 2.0;
            testingTime /= Math.pow(10, 9);
            String weights = "";
	    double weights_d[] = networks[i].getWeights();
            for (int x = 0; x < weights_d.length-1; x++)
            {
   		weights += df.format(weights_d[i]) + ",";
            }
   		weights += df.format(weights_d[weights_d.length-1]) + ",";
            results += df.format(incorrect / (correct + incorrect)) + ","
                     + df.format(trainingTime) + "," + df.format(testingTime) + "\n" + weights + "\n";

        }

        System.out.print(results);
    }

    private static Instance[] initializeInstances(String dataFile) {

        System.out.println(new File("").getAbsolutePath() + "/src/opt/test/" + dataFile);
        DataSetReader dsr = new CSVDataSetReader(new File("").getAbsolutePath() + "/src/opt/test/" + dataFile);
        DataSet ds;
        DataSet labs;

        try {
            ds = dsr.read(1);
            return ds.getInstances();
        } catch (Exception e) {
            System.out.println("Failed to read input file");
            return null;
        }
    }
}
