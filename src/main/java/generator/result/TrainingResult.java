package generator.result;

import generator.TrainingSettings;
import statistic.TrainingStatistics;

/**
 *
 * @author Milan
 */
public class TrainingResult {

    private double totalError;
    private int iterations;
    
    private TrainingStatistics MSE;
    private TrainingStatistics iterationStat;
    private boolean learned;
    private TrainingSettings settings;

    public TrainingResult(TrainingSettings settings, double totalError, int iterations) {
        this.totalError = totalError;
        this.iterations = iterations;
        this.settings = settings;
    }

    public TrainingResult(TrainingSettings ts) {
        this.settings = ts;
    }

    public int getIterations() {
        return iterations;
    }

    public void setIterations(int iterations) {
        this.iterations = iterations;
    }

    public boolean isLearned() {
        return learned;
    }

    public void setLearned(boolean learned) {
        this.learned = learned;
    }

    public double getTotalError() {
        return totalError;
    }

    public void setTotalError(double totalError) {
        this.totalError = totalError;
    }

    public TrainingSettings getSettings() {
        return settings;
    }

    public void setSettings(TrainingSettings settings) {
        this.settings = settings;
    }

    public TrainingStatistics getMSE() {
        return MSE;
    }

    public void setMSE(TrainingStatistics MSE) {
        this.MSE = MSE;
    }

    public TrainingStatistics getIterationStat() {
        return iterationStat;
    }

    public void setIterationStat(TrainingStatistics iterationStat) {
        this.iterationStat = iterationStat;
    }

}
