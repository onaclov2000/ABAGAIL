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
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class ContinuousPeaksTest {
    /** The n value */
    private static final int N = 60;
    /** The t value */
    private static final int T = N / 10;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        	Random random = new Random(6);
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new ContinuousPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        FixedIterationTrainer fit = null;
        double result[] = new double[100];
        double runtime[] = new double[100];
        for (int n=0; n<100; n++){
         RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp); 
         System.out.println("Starting Point: " + ef.value(rhc.getOptimal()));
         for (int i=1; i<100; i++){
	     double start = System.nanoTime();     
             fit = new FixedIterationTrainer(rhc, i * 200); // original 200000
            fit.train();
             result[i] = ef.value(rhc.getOptimal());
	     double end = System.nanoTime();
	     runtime[i] = (end - start)  / Math.pow(10,9);
        }
        System.out.println("RHC: " + Arrays.toString(result));
        System.out.println("Runtime: " + Arrays.toString(runtime));
        }
        
        double temperature[] = new double[500];
        double cooling[] = new double[500];
        for (int n = 0; n < 100; n++){
             temperature[n] = random.nextDouble() * 2E12;
             cooling[n] = random.nextDouble();
             SimulatedAnnealing sa = new SimulatedAnnealing(temperature[n], cooling[n], hcp);
         for (int i=1; i<100; i++){
	     double start = System.nanoTime();
             fit = new FixedIterationTrainer(sa, i * 200);
            result[i] = ef.value(sa.getOptimal());
	     double end = System.nanoTime();
	     runtime[i] = (end - start)  / Math.pow(10,9);
         }
         System.out.println("SA: " + Arrays.toString(result));
         System.out.println("T: " + temperature[n]);
         System.out.println("C: " + cooling[n]);
         System.out.println("Runtime: " + Arrays.toString(runtime));
        }
         int populationSize[] = new int[100];
         int toMate[] = new int[100];
         int toMutate[] = new int[100];
        for (int n = 0; n<100; n++){
    //int populationSize, int toMate, int toMutate
             populationSize[n] = random.nextInt(500) + 2;
             toMate[n] = random.nextInt(populationSize[n]);
             toMutate[n] = random.nextInt(10);
             StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(populationSize[n], toMate[n], toMutate[n], gap);
         for (int i=1; i<100; i++){
    	     double start = System.nanoTime();
             fit = new FixedIterationTrainer(ga, i * 10);
             fit.train();
             result[i] = ef.value(ga.getOptimal());
	         double end = System.nanoTime();
	         runtime[i] = (end - start)  / Math.pow(10,9);
         }
         System.out.println("GA: " + Arrays.toString(result));
         System.out.println("populationSize: " + populationSize[n]);
         System.out.println("toMate: " + toMate[n]);
         System.out.println("toMutate: " + toMutate[n]);
  	 System.out.println("Runtime: " + Arrays.toString(runtime));
        }
        
        int samples[] = new int[100];
	int tokeep[] = new int[100];
         for (int n = 0; n<100; n++){
           samples[n] = random.nextInt(200);
           if (samples[n] <= 0){
                samples[n] = samples[n] * -1;
            }
            tokeep[n] = random.nextInt(samples[n]-1);
            MIMIC mimic = new MIMIC(samples[n], tokeep[n], pop);
        // int samples, int tokeep
	    for (int i = 1; i < 100; i++){
		    double start = System.nanoTime();
  	            fit = new FixedIterationTrainer(mimic, i * 10);
     	            fit.train();
	            result[i] = ef.value(mimic.getOptimal());
		    double end = System.nanoTime();
		    runtime[i] = (end - start)  / Math.pow(10,9);
             }
  	System.out.println("MIMIC: " + Arrays.toString(result));
  	System.out.println("sample: " + samples[n]);
  	System.out.println("tokeep: " + tokeep[n]);
  	System.out.println("Runtime: " + Arrays.toString(runtime));
	}

    }
}
