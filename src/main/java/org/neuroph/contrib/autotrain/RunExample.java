/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.neuroph.contrib.autotrain;

import java.io.FileNotFoundException;
import java.util.List;
import org.neuroph.core.data.DataSet;
import org.neuroph.util.TransferFunctionType;

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
        AutoTrainer trainer = new AutoTrainer()
                .setMaxError(0.01)
                .setMaxIterations(20000)
                .setTransferFunction(TransferFunctionType.TRAPEZOID)
                .setHiddenNeurons(new Range(10, 20),2)    // kako dodati jos slojeva neurona?
                .setLearningRate(new Range(0.3, 0.9),0.3)
                .repeat(5)
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
