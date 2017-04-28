/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.neuroph.contrib.autotrain;

import org.neuroph.contrib.autotrain.AutoTrainer;
import org.neuroph.contrib.autotrain.Range;
import org.neuroph.contrib.autotrain.TrainingResult;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.contrib.autotrain.Util;

/**
 * kako iskoristiti vrednosti iz matrice(TP, TN, FP, FN) kako tokom testiranja
 * videti broj pogodjenih, broj promasenih Memento?
 *
 */
/**
 *
 * @author Milan
 */
public class RunExample {

    private static final String FILEPATH = "Iris/Iris-dataset-normalised.txt";

    public static void main(String[] args) {
        // maxError, maxIterations i to u TrainingSettings
        //dodati i transfer function type: sigmoid, Tanh, 
        AutoTrainer trainer = new AutoTrainer()
                .setMaxError(0.01)
                .setMaxIterations(20000)
                //.setTransferFunction(TransferFunctionType.TANH) ---- nece da radi sa transfer funkcijom
                .setHiddenNeurons(new Range(20, 23))    // kako dodati jos slojeva neurona?
                .setLearningRate(new Range(0.3, 0.5))
                .repeat(3)
                .setTrainTestSplit(70);

        DataSet dataSet = DataSet.createFromFile(FILEPATH, 4, 3, "\t", true);
        trainer.train(dataSet);
        List<TrainingResult> results = trainer.getResults();

        try {
            Util.saveToCSV(trainer, results);
        } catch (FileNotFoundException ex) {
            System.out.println("Error writing csv file");
        }

        System.out.println("Main done!");
    }
}
