package shared;
import opt.EvaluationFunction;
import opt.OptimizationAlgorithm;
/**
 * A fixed iteration trainer
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FixedIterationTrainer implements Trainer {
    
    /**
     * The inner trainer
     */
    private Trainer trainer;
    
    /**
     * The number of iterations to train
     */
    private int iterations;
    
    private EvaluationFunction eval;
    
    /**
     * Make a new fixed iterations trainer
     * @param t the trainer
     * @param iter the number of iterations
     */
    public FixedIterationTrainer(Trainer t, int iter, EvaluationFunction ef) {
        trainer = t;
        iterations = iter;
        eval = ef;
    }

    /**
     * @see shared.Trainer#train()
     */
    public double train() {
        double sum = 0;
        for (int i = 0; i < iterations; i++) {
            sum += trainer.train();
        }
        return sum / iterations;
    }
     public int train(int opt) {
        double sum = 0;
        int i = 0;
        int current_val = 0;
        double previous_opt = 0.0;
        OptimizationAlgorithm temp = (OptimizationAlgorithm) trainer;
        while ((int)eval.value(temp.getOptimal()) < opt && current_val < 2000){
                i++;
                if (eval.value(temp.getOptimal()) == previous_opt){
                    current_val++;
                }
                else{
                    current_val = 0;    
                }
                previous_opt = eval.value(temp.getOptimal());
                sum += trainer.train();
                temp = (OptimizationAlgorithm) trainer;
        }
        if (current_val == 2000){
            return i - current_val;
        }
        return i;
    }
    

}
