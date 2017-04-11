package generator;

/**
 *
 * @author Milan
 */
public class TrainingSettings {
    
    private double learningRate;
    private double momentum;
    private int hiddenNeurons;

// procenat za trening i test
    private int trainingSet;
    private int testSet;
    
    
    
    public TrainingSettings(double learningRate, double momentum, int hiddenNeurons) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.hiddenNeurons = hiddenNeurons;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getMomentum() {
        return momentum;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public int getHiddenNeurons() {
        return hiddenNeurons;
    }

    public void setHiddenNeurons(int hiddenNeurons) {
        this.hiddenNeurons = hiddenNeurons;
    }

    public int getTrainingSet() {
        return trainingSet;
    }

    public void setTrainingSet(int trainingSet) {
        this.trainingSet = trainingSet;
    }

    public int getTestSet() {
        return testSet;
    }

    public void setTestSet(int testSet) {
        this.testSet = testSet;
    }
    
    
}
