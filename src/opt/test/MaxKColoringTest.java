package opt.test;

import java.util.Arrays;
import java.util.Random;

import opt.ga.MaxKColorFitnessFunction;
import opt.ga.Vertex;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author kmandal
 * @version 1.0
 */
public class MaxKColoringTest {
    /** The n value */
    private static final int N = 50; // number of vertices
    private static final int L =4; // L adjacent nodes per vertex
    private static final int K = 8; // K possible colors
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        Random random = new Random(N*L);
        // create the random velocity
        Vertex[] vertices = new Vertex[N];
        for (int i = 0; i < N; i++) {
            Vertex vertex = new Vertex();
            vertices[i] = vertex;	
            vertex.setAdjMatrixSize(L);
            for(int j = 0; j <L; j++ ){
            	 vertex.getAadjacencyColorMatrix().add(random.nextInt(N*L));
            }
        }
        /*for (int i = 0; i < N; i++) {
            Vertex vertex = vertices[i];
            System.out.println(Arrays.toString(vertex.getAadjacencyColorMatrix().toArray()));
        }*/
        // for rhc, sa, and ga we use a permutation based encoding
        MaxKColorFitnessFunction ef = new MaxKColorFitnessFunction(vertices);
        Distribution odd = new DiscretePermutationDistribution(K);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new SingleCrossOver();
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        
        Distribution df = new DiscreteDependencyTree(.1); 
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        

        FixedIterationTrainer fit = null;
        double result[] = new double[100];
        double runtime[] = new double[100];
        boolean conflict[] = new boolean[100];
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
	     conflict[i] = ef.foundConflict(1);
        }
        System.out.println("RHC: " + Arrays.toString(result));
        System.out.println("Runtime: " + Arrays.toString(runtime));
        System.out.println("Conflicts: " + Arrays.toString(conflict));
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
             conflict[i] = ef.foundConflict(1);
         }
         System.out.println("SA: " + Arrays.toString(result));
         System.out.println("T: " + temperature[n]);
         System.out.println("C: " + cooling[n]);
         System.out.println("Runtime: " + Arrays.toString(runtime));
         System.out.println("Conflicts: " + Arrays.toString(conflict));
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
             fit = new FixedIterationTrainer(ga, i * 5);
             fit.train();
             result[i] = ef.value(ga.getOptimal());
	     double end = System.nanoTime();
	     runtime[i] = (end - start)  / Math.pow(10,9);
             conflict[i] = ef.foundConflict(1);
         }
         System.out.println("GA: " + Arrays.toString(result));
         System.out.println("populationSize: " + populationSize[n]);
         System.out.println("toMate: " + toMate[n]);
         System.out.println("toMutate: " + toMutate[n]);
  	 System.out.println("Runtime: " + Arrays.toString(runtime));
         System.out.println("Conflicts: " + Arrays.toString(conflict));
        }

        int samples[] = new int[100];
	int tokeep[] = new int[100];
	int trainCount[] = new int [100];
         for (int n = 0; n<100; n++){
           samples[n] = random.nextInt(200);
           if (samples[n] <= 0){
                samples[n] = samples[n] * -1;
            }
            tokeep[n] = random.nextInt(samples[n]-1);
            //System.out.println("n: " + n);
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
        System.out.println("Conflicts: " + Arrays.toString(conflict));
	}
    }
}
