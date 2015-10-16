package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
    private static final int N = 50;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();   
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        
        FixedIterationTrainer fit = null;
        double result[] = new double[100];
	double runtime[] = new double[100];

        for (int i=0; i<100; i++){
	    double start = System.nanoTime();
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
            fit = new FixedIterationTrainer(rhc, 200000);
            fit.train();
            result[i] = ef.value(rhc.getOptimal());
	    double end = System.nanoTime();
	    runtime[i] = (end - start)  / Math.pow(10,9);
        }
        System.out.println("RHC: " + Arrays.toString(result));
        System.out.println("Runtime: " + Arrays.toString(runtime));

        double temperature[] = new double[100];
        double cooling[] = new double[100];
        for (int i=0; i<100; i++){
	    double start = System.nanoTime();
            temperature[i] = random.nextDouble() * 2E12;
            cooling[i] = random.nextDouble();
            SimulatedAnnealing sa = new SimulatedAnnealing(temperature[i], cooling[i], hcp);
            fit = new FixedIterationTrainer(sa, 200000);
            fit.train();
            result[i] = ef.value(sa.getOptimal());
	    double end = System.nanoTime();
	    runtime[i] = (end - start)  / Math.pow(10,9);
        }
        System.out.println("SA: " + Arrays.toString(result));
        System.out.println("T: " + Arrays.toString(temperature));
        System.out.println("C: " + Arrays.toString(cooling));
        System.out.println("Runtime: " + Arrays.toString(runtime));
        
        int populationSize[] = new int[100];
        int toMate[] = new int[100];
        int toMutate[] = new int[100];
        
        for (int i=0; i<100; i++){
	    double start = System.nanoTime();
            //int populationSize, int toMate, int toMutate
            populationSize[i] = random.nextInt(500) + 2;
            toMate[i] = random.nextInt(populationSize[i]);
            toMutate[i] = random.nextInt(10);
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(populationSize[i], toMate[i], toMutate[i], gap);
            fit = new FixedIterationTrainer(ga, 1000);
            fit.train();
            result[i] = ef.value(ga.getOptimal());
	    double end = System.nanoTime();
	    runtime[i] = (end - start)  / Math.pow(10,9);
        }
        System.out.println("GA: " + Arrays.toString(result));
        System.out.println("populationSize: " + Arrays.toString(populationSize));
        System.out.println("toMate: " + Arrays.toString(toMate));
        System.out.println("toMutate: " + Arrays.toString(toMutate));
  	System.out.println("Runtime: " + Arrays.toString(runtime));

        // for mimic we use a sort encoding
        ef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
       
        int samples[] = new int[100];
	int tokeep[] = new int[100];
	
        // int samples, int tokeep
	for (int i = 0; i < 100; i++){
double start = System.nanoTime();
		samples[i] = random.nextInt();
		tokeep[i] = random.nextInt(samples[i] + 1);
	        MIMIC mimic = new MIMIC(samples[i], tokeep[i], pop);        
fit = new FixedIterationTrainer(mimic, 1000);
            fit.train();
            result[i] = ef.value(mimic.getOptimal());
		double end = System.nanoTime();
		runtime[i] = (end - start)  / Math.pow(10,9);
	}
  	System.out.println("MIMIC: " + Arrays.toString(result));
  	System.out.println("samples: " + Arrays.toString(samples));
  	System.out.println("tokeep: " + Arrays.toString(tokeep));
  	System.out.println("Runtime: " + Arrays.toString(runtime));   
    }
}
