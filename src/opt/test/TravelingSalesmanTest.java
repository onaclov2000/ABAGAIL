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
        Random random = new Random(5000);
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
        double results[] = new double[100];
        
        System.out.println("RandomizedHillClimbing Results with 100 restarts");
        for (int i=0; i < 100; i++){
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp); 
            fit = new FixedIterationTrainer(rhc, 200000);
            fit.train();
            results[i] = ef.value(rhc.getOptimal());
        }
        System.out.println(Arrays.toString(results));
        System.out.println("Simulated Annealing Results");
        
        double temperature[] = new double[100];
        double cooling[] = new double[100];
        for (int i=0; i < 100; i++){
            temperature[i] = random.nextDouble() * 2000000000000.0;
            cooling[i] = random.nextDouble();
            SimulatedAnnealing sa = new SimulatedAnnealing(temperature[i], cooling[i], hcp);
            fit = new FixedIterationTrainer(sa, 200000);
            fit.train();
            results[i] = ef.value(sa.getOptimal());
            
        }
        System.out.println(Arrays.toString(results));
        System.out.println(Arrays.toString(temperature));
        System.out.println(Arrays.toString(cooling));
        
        System.out.println("StandardGeneticAlgorithm Results");
        for (int i=0; i < 100; i++){
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
            fit = new FixedIterationTrainer(ga, 1000);
            fit.train();
            results[i] = ef.value(ga.getOptimal());
        }
        System.out.println(Arrays.toString(results));
        
        // for mimic we use a sort encoding
        ef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        System.out.println("MIMIC Results");
        MIMIC mimic = new MIMIC(200, 100, pop);
        fit = new FixedIterationTrainer(mimic, 1000);
        fit.train();
        System.out.println(ef.value(mimic.getOptimal()));
        
    }
}
