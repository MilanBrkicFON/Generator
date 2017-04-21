package generator;

import generator.result.TrainingResult;
import java.util.ArrayList;
import java.util.List;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import statistic.TrainingStatistics;

/**
 *
 * @author Milan
 */
public class AutoTrainer implements Trainer {

    private List<TrainingSettings> trainingSettingsList;

    private List<TrainingResult> results;

    // Range(min, Max)
    private int maxHiddenNeurons;
    private double maxLearningRate;
    private double maxMomentum = 0.9;

    private double minLearningRate;
    private int minHiddenNeurons;
    private int splitPercentage = 100;
    private boolean splitTrainTest = false;

    private boolean generateStatistics = false;
    private int repeat = 1;

    /**
     *
     */
    public AutoTrainer() {
        trainingSettingsList = new ArrayList<>();
        results = new ArrayList<>();
    }

    /**
     * Get results.
     *
     * @return List of TrainingResult. If nothing was trained, method returns
     * empty ArrayList().
     *
     */
    public List<TrainingResult> getResults() {
        return results;
    }

    /**
     * Set range for hidden neurons of neural network. Auto trainer is looping
     * through that range.
     *
     * @param range given for hidden neurons
     */
    public AutoTrainer setHiddenNeurons(Range range) {
        this.minHiddenNeurons = (int) range.getMin();
        this.maxHiddenNeurons = (int) range.getMax();
        return this;
    }

    /**
     * Set range for learning rate of neural network. Auto trainer is looping
     * through that range.
     *
     * @param range given for learning rate
     */
    public AutoTrainer setLearningRate(Range range) {
        this.minLearningRate = range.getMin();
        this.maxLearningRate = range.getMax();
        return this;
    }

    /**
     *
     * @param maxMomentum
     */
    public void setMaxMomentum(double maxMomentum) {
        this.maxMomentum = maxMomentum;
    }


    /**
     * Repeat neural network with same parameters specified number of times and create statistic.
     *
     * @param times to repeat network
     */
    public AutoTrainer repeat(int times) {
        this.repeat = times;
        generateStatistics = true;
        return this;
    }

    /**
     *
     * @return if statistic is enabled
     */
    public boolean generatesStatistics() {
        return generateStatistics;
    }

    /**
     * Set percentage of training set (in percents).
     *
     * @param trainingPrecent new value of splitPercentage
     */
    public AutoTrainer setTrainTestSplit(int trainingPrecent) {
        this.splitPercentage = trainingPrecent;
        this.splitTrainTest = true;
        return this;
    }

    /**
     *
     * @return true if split
     */
    public boolean isSplitForTesting() {
        return splitTrainTest;
    }

    private void generateTrainingSettings() {
        double pom = minLearningRate;
        while (minHiddenNeurons <= maxHiddenNeurons) {
            while (minLearningRate <= maxLearningRate) {
                //MOMENTUM for (double momentum = 0.1; momentum < maxMomentum; momentum += 0.1) { proveriti za sta je potreban momentum i kako se koristi!
                TrainingSettings ts = new TrainingSettings(minLearningRate, 0.7, minHiddenNeurons);
                this.trainingSettingsList.add(ts);
                minLearningRate += 0.1;
                //}
            }
            minLearningRate = pom;
            minHiddenNeurons++;
        }
        System.out.println("Generated : " + this.trainingSettingsList.size() + " settings.");
    }

    @Override
    public void train(NeuralNetwork neuralNet, DataSet dataSet) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }    
    
    /**
     *
     * You can get results calling getResults() method.
     *
     * @param neuralNetwork type of neural net
     * @param dataSet 
     */
    @Override
    public void train(DataSet dataSet) {// mozda da se vrati Training setting koji je najbolje resenje za dati dataset.??
        generateTrainingSettings();
        List<TrainingResult> statResults = null;
        DataSet trainingSet, testSet; // validationSet;

        // dataSet.split(sizePercents)
        // 1. ako nema samplinga : trening set = dataset. testSet = dataset
        // 2. ako ima samplnga : trening set = % dataseta. test set = % dataseta
        // 3. idradi proceduru treninga sa training setom
        // 4. odradi testiranje sa test setom
        if (splitTrainTest) {
            DataSet[] dataSplit = dataSet.sample(splitPercentage); //opet ne radi Maven za neuroph 2.92
            trainingSet = dataSplit[0];
            testSet = dataSplit[1];
           // System.out.println("Data set splited: Training set " + splitPercentage + ", Test set " + (100 - splitPercentage));
        } else { // use entire dataset for training and testing
            trainingSet = dataSet;
            testSet = dataSet;
        }

        if (generateStatistics) {
            statResults = new ArrayList<>();
        }

        int trainingNo = 0;
        for (TrainingSettings ts : trainingSettingsList) {
            System.out.println("-----------------------------------------------------------------------------------");
            trainingNo++;
            System.out.println("##TRAINING: " + trainingNo);
            ts.setTrainingSet(splitPercentage);
            ts.setTestSet(100 - splitPercentage);
            int subtrainNo = 0;

            do {
               subtrainNo++;
               System.out.println("#SubTraining: " + subtrainNo);
                        
                MultiLayerPerceptron neuralNet
                        = new MultiLayerPerceptron(dataSet.getInputSize(), ts.getHiddenNeurons(), dataSet.getOutputSize());

                BackPropagation bp = neuralNet.getLearningRule();

                bp.setLearningRate(ts.getLearningRate());
                bp.setMaxError(0.001);
                bp.setMaxIterations(20000);

                neuralNet.learn(dataSet);

                testNeuralNetwork(neuralNet, testSet);
                TrainingResult result = new TrainingResult(ts, bp.getTotalNetworkError(), bp.getCurrentIteration());
                System.out.println(subtrainNo + ") " + bp.getCurrentIteration());
            
                if (generateStatistics) {
                    statResults.add(result);
                } else {
                    results.add(result);
                }

            } while (subtrainNo < repeat);
            
            if (generateStatistics) {
                TrainingResult trainingStats = calculateTrainingStatistics(ts, statResults);
                results.add(trainingStats);
                statResults.clear();
//                System.out.println("Done statistic: #Training: " + trainingNo);
            }

        }

    }

    private TrainingResult calculateTrainingStatistics(TrainingSettings ts, List<TrainingResult> results) {
        System.out.println("working on statistic...");
        TrainingResult result = new TrainingResult(ts);

        TrainingStatistics iterationsStat = TrainingStatistics.calculateIterations(results);
        TrainingStatistics MSEStat = TrainingStatistics.calculateMSE(results);

        result.setMSE(MSEStat);
        result.setIterationStat(iterationsStat);

        return result;
    }

    private void testNeuralNetwork(MultiLayerPerceptron neuralNet, DataSet testingSet) {
        for (DataSetRow testSetRow : testingSet.getRows()) {
            neuralNet.setInput(testSetRow.getInput());
            neuralNet.calculate();
            double[] networkOutput = neuralNet.getOutput();
//
//            System.out.print("Input: " + Arrays.toString(testSetRow.getInput()));
//            System.out.println(" Output: " + Arrays.toString(networkOutput));
        }
    }


}
