/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package generator;

import generator.result.TrainingResult;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
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
    private boolean splitForTesting = false;

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
    private boolean statistics = false;

    /**
     * Repeat neural network with same parameters and create statistic.
     *
     * @param times to repeat network
     */
    public AutoTrainer repeatNetwork(int times) {
        this.repeat = times;
        statistics = true;
        return this;
    }

    /**
     *
     * @return if statistic is enabled
     */
    public boolean isStatistics() {
        return statistics;
    }

    /**
     * Set percentage of training set.
     *
     * @param trainingPrecent new value of splitPercentage
     */
    public AutoTrainer setSplitPercentage(int trainingPrecent) {
        this.splitPercentage = trainingPrecent;
        this.splitForTesting = true;
        return this;
    }

    /**
     *
     * @return true if split
     */
    public boolean isSplitForTesting() {
        return splitForTesting;
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

    /**
     *
     * You can get results calling getResults() method.
     *
     * @param neuralNetwork type of neural net
     * @param dataSet 
     */
    @Override
    public void train(NeuralNetwork neuralNetwork, DataSet dataSet) {// mozda da se vrati Training setting koji je najbolje resenje za dati dataset.??
        generateTrainingSettings();
        List<TrainingResult> StatResults = null;
        DataSet trainingSet, testSet; // validationSet;

        // dataSet.split(sizePercents)
        // 1. ako nema samplinga : trening set = dataset. testSet = dataset
        // 2. ako ima samplnga : trening set = % dataseta. test set = % dataseta
        // 3. idradi proceduru treninga sa training setom
        // 4. odradi testiranje sa test setom
        if (splitForTesting) {
            DataSet[] dataSplit = dataSet.sample(splitPercentage); //opet ne radi Maven za neuroph 2.92
            trainingSet = dataSplit[0];
            testSet = dataSplit[1];
            System.out.println("Data set splited: Training set " + splitPercentage + ", Test set " + (100 - splitPercentage));

        } else {
            trainingSet = dataSet;
            testSet = dataSet;
        }

        if (statistics) {
            //return trainNetworkMultipleTimes(dataSet);
            StatResults = new ArrayList<>();
        }

        //List<TrainingResult> trainingResultsList = new ArrayList<>();
        int i = 1;
        for (TrainingSettings ts : trainingSettingsList) {
            System.out.println("-----------------------------------------------------------------------------------");
            System.out.println("##TRAINING: " + i);
            ts.setTrainingSet(splitPercentage);
            ts.setTestSet(100 - splitPercentage);
            int j = 1;
            int pom = this.repeat;
            do {
                System.out.println("Subtrening: ");

                MultiLayerPerceptron neuralNet
                        = new MultiLayerPerceptron(dataSet.getInputSize(), ts.getHiddenNeurons(), dataSet.getOutputSize());

                BackPropagation bp = neuralNet.getLearningRule();

                bp.setLearningRate(ts.getLearningRate());
                bp.setMaxError(0.001);
                bp.setMaxIterations(20000);

                neuralNet.learn(dataSet);

                testNeuralNetwork(neuralNet, testSet);
                TrainingResult result
                        = new TrainingResult(ts, bp.getTotalNetworkError(), bp.getCurrentIteration());
                System.out.println(j + ") " + bp.getCurrentIteration());
                if (statistics) {
                    StatResults.add(result);
                    System.out.println("#SubTraining: " + j++);
                } else {
                    results.add(result);
                }

            } while (--pom > 0);
            if (statistics) {
                TrainingResult tr = doStatistic(ts, StatResults);
                results.add(tr);
                StatResults.clear();
                System.out.println("Done statistic: #Training: " + i);
            }
            i++;

        }

    }

    private TrainingResult doStatistic(TrainingSettings ts, List<TrainingResult> list) {
        System.out.println("working on statistic...");
        TrainingResult result = new TrainingResult(ts);

        List<TrainingResult> l = list;
        System.out.print("(" + l.size() + ")");
        TrainingStatistics iterationsStat = TrainingStatistics.calculateIterations(list);
        TrainingStatistics MSEStat = TrainingStatistics.calculateMSE(list);

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
