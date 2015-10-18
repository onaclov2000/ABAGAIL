package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;


/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int N = 200;
    /** The t value */
    private static final int T = N / 5;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
	Random random = new Random(6);
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        FixedIterationTrainer fit = null;
        double result[] = new double[30];
	    double runtime[] = new double[30];

        // for (int i=0; i<500; i++){
	    // double start = System.nanoTime();
            // RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
            // fit = new FixedIterationTrainer(rhc, 200000); // original 200000
            // fit.train();
            // result[i] = ef.value(rhc.getOptimal());
	    // double end = System.nanoTime();
	    // runtime[i] = (end - start)  / Math.pow(10,9);
        // }
        // System.out.println("RHC: " + Arrays.toString(result));
        // System.out.println("Runtime: " + Arrays.toString(runtime));

        // double temperature[] = new double[500];
        // double cooling[] = new double[500];
        // for (int i=0; i<500; i++){
	    // double start = System.nanoTime();
            // temperature[i] = random.nextDouble() * 2E12;
            // cooling[i] = random.nextDouble();
            // SimulatedAnnealing sa = new SimulatedAnnealing(temperature[i], cooling[i], hcp);
            // fit = new FixedIterationTrainer(sa, 200000);
            // fit.train();
            // result[i] = ef.value(sa.getOptimal());
	    // double end = System.nanoTime();
	    // runtime[i] = (end - start)  / Math.pow(10,9);
        // }
        // System.out.println("SA: " + Arrays.toString(result));
        // System.out.println("T: " + Arrays.toString(temperature));
        // System.out.println("C: " + Arrays.toString(cooling));
        // System.out.println("Runtime: " + Arrays.toString(runtime));
        
        // int populationSize[] = new int[500];
        // int toMate[] = new int[500];
        // int toMutate[] = new int[500];
        
        // for (int i=0; i<500; i++){
    	    // double start = System.nanoTime();
            // //int populationSize, int toMate, int toMutate
            // populationSize[i] = random.nextInt(500) + 2;
            // toMate[i] = random.nextInt(populationSize[i]);
            // toMutate[i] = random.nextInt(10);
            // StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(populationSize[i], toMate[i], toMutate[i], gap);
            // fit = new FixedIterationTrainer(ga, 1000);
            // fit.train();
            // result[i] = ef.value(ga.getOptimal());
	        // double end = System.nanoTime();
	        // runtime[i] = (end - start)  / Math.pow(10,9);
        // }
        // System.out.println("GA: " + Arrays.toString(result));
        // System.out.println("populationSize: " + Arrays.toString(populationSize));
        // System.out.println("toMate: " + Arrays.toString(toMate));
        // System.out.println("toMutate: " + Arrays.toString(toMutate));
  	    // System.out.println("Runtime: " + Arrays.toString(runtime));
        
    	int samples[] = new int[100];
	    int tokeep[] = new int[100];
	
        // int samples, int tokeep
	    for (int i = 1; i < 30; i++){
		    double start = System.nanoTime();	
           // samples[i] = random.nextInt(200);
		   // if (samples[i] <= 0){
           //     samples[i] = samples[i] * -1;
          //  }
          //  tokeep[i] = random.nextInt(samples[i]);
            MIMIC mimic = new MIMIC(200, 100, pop);
  	  	    fit = new FixedIterationTrainer(mimic, i * 10);
     	    fit.train();
		    result[i] = ef.value(mimic.getOptimal());
		    double end = System.nanoTime();
		    runtime[i] = (end - start)  / Math.pow(10,9);
    }
  	System.out.println("MIMIC: " + Arrays.toString(result));
  	//System.out.println("samples: " + Arrays.toString(samples));
  //	System.out.println("tokeep: " + Arrays.toString(tokeep));
  	System.out.println("Runtime: " + Arrays.toString(runtime));
		
    }
}

