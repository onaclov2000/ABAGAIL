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
    private static final int N = 200;
    /** The t value */
    private static final int T = N / 5;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
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
        double result[] = new double[100];
        double temperature[] = new double[100];
        double cooling[] = new double[100];
        for (int i=0; i<100; i++){
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
            fit = new FixedIterationTrainer(rhc, 200000); // original 200000
            fit.train();
            result[1] = ef.value(rhc.getOptimal());
        }
        System.out.println("RHC: " + Array.toString(result));
        for (int i=0; i<100; i++){
            temperature[i] = random.nextDouble() * 2E11;
            cooling[i] = random.nextDouble();
            SimulatedAnnealing sa = new SimulatedAnnealing(temperature[i], cooling[i], hcp);
            fit = new FixedIterationTrainer(sa, 200000);
            fit.train();
            result[i] = ef.value(sa.getOptimal());
        }
        System.out.println("SA: " + Array.toString(result));
        System.out.println("T: " + Array.toString(temperature));
        System.out.println("C: " + Array.toString(cooling));
        
        
        int populationSize[] = new int[100];
        int toMate[] = new int[100];
        int toMutate[] = new int[100];
        
        for (int i=0; i<100; i++){
            //int populationSize, int toMate, int toMutate
            populationSize[i] = random.nextInt(500);
            toMate[i] = random.nextInt(500);
            toMutate[i] = random.nextInt(50);
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(populationSize[i], toMate[i], toMutate[i], gap);
            fit = new FixedIterationTrainer(ga, 1000);
            fit.train();
            results[i] = ef.value(ga.getOptimal());
        }
        System.out.println("GA: " + Array.toString(results));
        System.out.println("populationSize: " + Array.toString(populationSize));
        System.out.println("toMate: " + Array.toString(toMate));
        System.out.println("toMutate: " + Array.toString(toMutate));
        
        
        MIMIC mimic = new MIMIC(200, 20, pop);
        fit = new FixedIterationTrainer(mimic, 1000);
        fit.train();
        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
    }
}

