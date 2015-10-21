package opt.test;

import java.util.Arrays;

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
   
    /** The t value */
    
    
    public static void main(String[] args) {
        int N = args.length > 0 ? Integer.parseInt(args[0]): 200;
        int T = N / 5;
        int[] ranges = new int[N];
        System.out.println("N: " + N);
        int global_optimum = (N-(T + 1) + N);
        System.out.println("Global Optimum: " + global_optimum);
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
        int i = 0;
        double start, end, trainingTime;
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp); 
        start= System.nanoTime();
        fit = new FixedIterationTrainer(rhc, i, ef);
        i = fit.train(global_optimum);
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);
        System.out.println("RHC: " + ef.value(rhc.getOptimal()) + "," + i + ',' + trainingTime);
        
        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
        start= System.nanoTime();
        fit = new FixedIterationTrainer(sa, i, ef);
        i = fit.train(global_optimum);
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);
        System.out.println("SA: " + ef.value(sa.getOptimal()) + "," + i + ',' + trainingTime);
        

        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
        start= System.nanoTime();
        fit = new FixedIterationTrainer(ga, i, ef);
        i = fit.train(global_optimum);
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);
        System.out.println("GA: " + ef.value(ga.getOptimal()) + "," + i + ',' + trainingTime);
        
        MIMIC mimic = new MIMIC(200, 20, pop);
        start= System.nanoTime();
        fit = new FixedIterationTrainer(mimic, i, ef);
        i = fit.train(global_optimum);
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);
        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()) + "," + i + ',' + trainingTime);
    }
}
