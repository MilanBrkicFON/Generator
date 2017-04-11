/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import generator.AutoTrainer;
import generator.Range;
import generator.result.TrainingResult;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import utility.Util;

/**
 * kako iskoristiti vrednosti iz matrice(TP, TN, FP, FN) kako tokom testiranja
 * videti broj pogodjenih, broj promasenih Memento?
 *
 */
/**
 *
 * @author Milan
 */
public class Main {

    private static final String FILEPATH = "Iris/Iris-dataset-normalised.txt";

    public static void main(String[] args) {
        AutoTrainer trainer = new AutoTrainer()
                .setHiddenNeurons(new Range(13, 15))
                .setLearningRate(new Range(0.3, 0.5))
                .repeatNetwork(3)
                .setSplitPercentage(70);

        DataSet dataSet = DataSet.createFromFile(FILEPATH, 4, 3, "\t", true);
        trainer.train(new NeuralNetwork(), dataSet);
        List<TrainingResult> results = trainer.getResults();

        try {
            Util.saveToCSV(trainer, results);
        } catch (FileNotFoundException ex) {
            System.out.println("Error writing csv file");
        }

        System.out.println("Main done!");
    }
}
