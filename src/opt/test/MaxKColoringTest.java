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
        double conflict[] = new double[100];
	    double runtime[] = new double[100];

        for (int i=0; i<100; i++){
	    double start = System.nanoTime();
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
            fit = new FixedIterationTrainer(rhc, 20000);
            fit.train();
            result[i] = ef.value(rhc.getOptimal());
            conflict[i] = ef.foundConflict();
            runtime[i] = (end - start)  / Math.pow(10,9);
        }
        System.out.println("RHC: " + Arrays.toString(result));
        System.out.println("Conflict: " + Arrays.toString(conflict));
        System.out.println("Runtime: " + Arrays.toString(runtime));

        System.out.println("============================");
        
        starttime = System.currentTimeMillis();
        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .1, hcp);
        fit = new FixedIterationTrainer(sa, 20000);
        fit.train();
        System.out.println("SA: " + ef.value(sa.getOptimal()));
        System.out.println(ef.foundConflict());
        System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
        
        System.out.println("============================");
        
        double temperature[] = new double[100];
        double cooling[] = new double[100];
        for (int i=0; i<100; i++){
	        double start = System.nanoTime();
            temperature[i] = random.nextDouble() * 2E12;
            cooling[i] = random.nextDouble();
            SimulatedAnnealing sa = new SimulatedAnnealing(temperature[i], cooling[i], hcp);
            fit = new FixedIterationTrainer(ga, 50);
            fit.train();
            result[i] = ef.value(sa.getOptimal());
	        double end = System.nanoTime();
            runtime[i] = (end - start)  / Math.pow(10,9);
            conflict[i] = ef.foundConflict();
        }
        System.out.println("SA: " + Arrays.toString(result));
        System.out.println("T: " + Arrays.toString(temperature));
        System.out.println("C: " + Arrays.toString(cooling));
        System.out.println("Runtime: " + Arrays.toString(runtime));
        System.out.println("Conflict: " + Arrays.toString(conflict));
        System.out.println("Time : "+ (System.currentTimeMillis() - starttime));
        System.out.println("============================");
        
        int samples[] = new int[100];
	    int tokeep[] = new int[100];
	
        // int samples, int tokeep
        for (int i = 0; i < 100; i++){
		    double start = System.nanoTime();
		    samples[i] = random.nextInt();
		    tokeep[i] = random.nextInt(samples[i] + 1);
	        MIMIC mimic = new MIMIC(samples[i], tokeep[i], pop);
            fit = new FixedIterationTrainer(mimic, 5);
            fit.train();
            result[i] = ef.value(mimic.getOptimal());
	        double end = System.nanoTime();
	        runtime[i] = (end - start)  / Math.pow(10,9);
	        conflict[i] = ef.foundConflict();

	}
  	System.out.println("MIMIC: " + Arrays.toString(result));
  	System.out.println("samples: " + Arrays.toString(samples));
  	System.out.println("tokeep: " + Arrays.toString(tokeep));
  	System.out.println("Runtime: " + Arrays.toString(runtime));
    System.out.println("Conflict: " + Arrays.toString(conflict));

        
    }
}
