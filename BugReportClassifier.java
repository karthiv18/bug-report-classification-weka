import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.stopwords.Rainbow;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;
import java.util.Random;

/**
 * BugReportClassifier.java
 *
 * Classifies bug reports from the Caffe GitHub repository as bug (1) or
 * non-bug (0), using TF-IDF features on the Title + Body text.
 *
 * Dataset: caffe_clean.csv  (286 instances: 33 bugs, 253 non-bugs)
 * Class imbalance ratio ~1:8 — macro F1 is the primary metric, as it
 * treats both classes equally regardless of frequency.
 *
 * Classifiers evaluated:
 *   1. Naive Bayes  — baseline (as specified in the coursework)
 *   2. Random Forest — proposed solution
 *   3. SMO (SVM)    — second proposed comparator
 *
 * TF-IDF is applied INSIDE FilteredClassifier so it is fit on training
 * folds only — no data leakage into test folds.
 *
 * Usage:
 *   javac -cp ".:weka.jar" BugReportClassifier.java
 *   java  -cp ".:weka.jar" BugReportClassifier caffe_clean.csv
 */
public class BugReportClassifier {

    // --- Configuration (document these in the Setup section of your report) ---
    private static final int NUM_CV_FOLDS = 10;
    private static final int RANDOM_SEED  = 42;
    private static final int MAX_WORDS    = 1000; // top-N tokens after TF-IDF
    private static final int RF_TREES     = 100;
    private static final int RF_FEATURES  = 32;   // ~sqrt(1000); justified below

    public static void main(String[] args) throws Exception {

        if (args.length < 1) {
            System.out.println("Usage: java -cp \".:weka.jar\" BugReportClassifier <caffe_clean.csv>");
            System.exit(1);
        }

        System.out.println("=== Bug Report Classifier — Caffe Dataset ===");
        System.out.println("Dataset     : " + args[0]);
        System.out.println("CV folds    : " + NUM_CV_FOLDS);
        System.out.println("Max words   : " + MAX_WORDS);
        System.out.println("RF trees    : " + RF_TREES);
        System.out.println("RF features : " + RF_FEATURES + "  (~sqrt(" + MAX_WORDS + "))");
        System.out.println();

        Instances rawData = loadCSV(args[0]);
        System.out.println("Instances   : " + rawData.numInstances());
        System.out.println("Classes     : " + rawData.numClasses()
                + " " + classDistribution(rawData));
        System.out.println("Note: Dataset is imbalanced — macro F1 is the primary metric.\n");

        // Evaluate baseline and proposed classifiers
        System.out.println("=== " + NUM_CV_FOLDS + "-fold Cross-Validation Results ===\n");
        runEvaluation("Naive Bayes  (Baseline)",   buildPipeline(new NaiveBayes()),    rawData);
        runEvaluation("Random Forest (Proposed)",  buildPipeline(buildRandomForest()), rawData);
        runEvaluation("SVM / SMO    (Proposed)",   buildPipeline(new SMO()),           rawData);
    }

    // ---------------------------------------------------------------
    // FilteredClassifier: wraps TF-IDF filter with a base classifier.
    // Weka re-fits StringToWordVector on each training fold — no leakage.
    // ---------------------------------------------------------------
    private static FilteredClassifier buildPipeline(Classifier base) throws Exception {
        StringToWordVector filter = new StringToWordVector();
        filter.setWordsToKeep(MAX_WORDS);
        filter.setTFTransform(true);
        filter.setIDFTransform(true);
        filter.setLowerCaseTokens(true);
        filter.setStopwordsHandler(new Rainbow());
        filter.setOutputWordCounts(true);

        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(filter);
        fc.setClassifier(base);
        return fc;
    }

    // ---------------------------------------------------------------
    // Random Forest hyperparameters.
    // RF_FEATURES = 32 ≈ sqrt(MAX_WORDS=1000): standard heuristic for
    // classification tasks (Breiman, 2001); limits correlation between
    // trees while retaining enough signal per split.
    // ---------------------------------------------------------------
    private static RandomForest buildRandomForest() {
        RandomForest rf = new RandomForest();
        rf.setNumIterations(RF_TREES);
        rf.setNumFeatures(RF_FEATURES);
        rf.setMaxDepth(0);         // unlimited — let trees grow fully
        rf.setSeed(RANDOM_SEED);
        return rf;
    }

    // ---------------------------------------------------------------
    // Load CSV; class label must be the last column
    // ---------------------------------------------------------------
    private static Instances loadCSV(String path) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(path));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    // ---------------------------------------------------------------
    // Cross-validation and results output
    // ---------------------------------------------------------------
    private static void runEvaluation(String name, Classifier clf, Instances data)
            throws Exception {

        System.out.println("--- " + name + " ---");
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(clf, data, NUM_CV_FOLDS, new Random(RANDOM_SEED));

        double macroF1    = macroF1(eval, data);
        double weightedF1 = eval.weightedFMeasure();

        // Primary metric first (macro F1 — treats both classes equally)
        System.out.printf("  Macro F1        : %.4f   <-- primary metric%n", macroF1);
        System.out.printf("  Weighted F1     : %.4f%n",   weightedF1);
        System.out.printf("  Accuracy        : %.4f%%%n", eval.pctCorrect());
        System.out.printf("  Kappa           : %.4f%n",   eval.kappa());
        System.out.printf("  AUC (weighted)  : %.4f%n",   eval.weightedAreaUnderROC());

        // Per-class: critical for imbalanced data — check recall on minority 'bug' class
        System.out.println("\n  Per-class results (important: check 'bug' recall):");
        for (int i = 0; i < data.numClasses(); i++) {
            System.out.printf("    [%-9s]  Precision=%.4f  Recall=%.4f  F1=%.4f%n",
                    data.classAttribute().value(i),
                    eval.precision(i),
                    eval.recall(i),
                    eval.fMeasure(i));
        }

        // Confusion matrix
        System.out.println("\n  Confusion Matrix:");
        double[][] cm = eval.confusionMatrix();
        System.out.print("           ");
        for (int j = 0; j < data.numClasses(); j++) {
            System.out.printf("  %-10s", data.classAttribute().value(j));
        }
        System.out.println();
        for (int i = 0; i < cm.length; i++) {
            System.out.printf("  %-9s", data.classAttribute().value(i));
            for (double v : cm[i]) {
                System.out.printf("  %-10.0f", v);
            }
            System.out.println();
        }

        System.out.println("\n" + "=".repeat(60) + "\n");
    }

    // ---------------------------------------------------------------
    // Macro F1: simple average across classes — not weighted by support.
    // Preferred when classes are imbalanced (33 bugs vs 253 non-bugs).
    // ---------------------------------------------------------------
    private static double macroF1(Evaluation eval, Instances data) throws Exception {
        double sum = 0;
        for (int i = 0; i < data.numClasses(); i++) {
            sum += eval.fMeasure(i);
        }
        return sum / data.numClasses();
    }

    // ---------------------------------------------------------------
    // Summarise class distribution for display
    // ---------------------------------------------------------------
    private static String classDistribution(Instances data) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < data.numClasses(); i++) {
            int count = 0;
            for (int j = 0; j < data.numInstances(); j++) {
                if ((int) data.instance(j).classValue() == i) count++;
            }
            sb.append(data.classAttribute().value(i)).append("=").append(count);
            if (i < data.numClasses() - 1) sb.append(", ");
        }
        return sb.append("]").toString();
    }
}
