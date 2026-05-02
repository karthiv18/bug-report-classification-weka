import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.stopwords.Rainbow;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;
import java.util.Random;

/**
 * StatisticalTest.java
 *
 * Compares Naive Bayes (baseline) vs Random Forest (proposed) on the Caffe
 * bug report dataset using the 5x2cv paired t-test (Dietterich, 1998).
 *
 * Metric: Macro F1 (primary metric, chosen because the dataset is imbalanced:
 * 33 bugs vs 253 non-bugs; macro F1 treats both classes equally).
 *
 * Test design:
 *   - 5 independent repetitions, each with a unique random seed
 *   - Each rep: stratified 50/50 split, train on half A / test on half B,
 *     then swap — giving two difference scores p_i(1) and p_i(2) per rep
 *   - t-statistic computed per Dietterich (1998), equation 3
 *   - df = 5, critical value = 2.571 (alpha=0.05, two-tailed)
 *
 * Reference:
 *   Dietterich, T.G. (1998). Approximate statistical tests for comparing
 *   supervised classification learning algorithms. Neural Computation, 10(7),
 *   1895–1923.
 *
 * Usage:
 *   javac -cp ".:weka.jar" StatisticalTest.java
 *   java  -cp ".:weka.jar" StatisticalTest caffe_clean.csv
 */
public class StatisticalTest {

    private static final int    NUM_REPS       = 5;
    private static final int    MAX_WORDS      = 1000;
    private static final int    RF_TREES       = 100;
    private static final int    RF_FEATURES    = 32;
    private static final double CRITICAL_VALUE = 2.571; // df=5, alpha=0.05, two-tailed

    public static void main(String[] args) throws Exception {

        if (args.length < 1) {
            System.out.println("Usage: java -cp \".:weka.jar\" StatisticalTest <caffe_clean.csv>");
            System.exit(1);
        }

        Instances rawData = loadCSV(args[0]);

        System.out.println("=== 5x2cv Paired t-Test (Dietterich, 1998) ===");
        System.out.println("Dataset  : " + args[0] + "  (" + rawData.numInstances() + " instances)");
        System.out.println("Compared : Naive Bayes (baseline) vs Random Forest (proposed)");
        System.out.println("Metric   : Macro F1  (primary; dataset is imbalanced ~1:8)");
        System.out.printf( "df=5, critical value (alpha=0.05, two-tailed): %.3f%n%n", CRITICAL_VALUE);

        double[] diff1 = new double[NUM_REPS]; // NB - RF on fold A
        double[] diff2 = new double[NUM_REPS]; // NB - RF on fold B (swapped)

        for (int i = 0; i < NUM_REPS; i++) {
            Random rng = new Random(i * 100L + 7);

            // Stratified 50/50 split
            rawData.randomize(rng);
            rawData.stratify(2);
            Instances foldA = rawData.testCV(2, 0);
            Instances foldB = rawData.testCV(2, 1);

            // Train on A, test on B
            double nbAB = evalHoldout(buildPipeline(new NaiveBayes()),    foldA, foldB);
            double rfAB = evalHoldout(buildPipeline(buildRandomForest()), foldA, foldB);
            diff1[i] = nbAB - rfAB;

            // Train on B, test on A (swap)
            double nbBA = evalHoldout(buildPipeline(new NaiveBayes()),    foldB, foldA);
            double rfBA = evalHoldout(buildPipeline(buildRandomForest()), foldB, foldA);
            diff2[i] = nbBA - rfBA;

            System.out.printf(
                "Rep %d  |  A→B: NB=%.4f RF=%.4f diff=%.4f  |  B→A: NB=%.4f RF=%.4f diff=%.4f%n",
                i + 1, nbAB, rfAB, diff1[i], nbBA, rfBA, diff2[i]);
        }

        double tStat   = computeTStatistic(diff1, diff2);
        double meanDiff = overallMeanDiff(diff1, diff2); // negative = RF is better

        System.out.printf("%nt-statistic          : %.4f%n", tStat);
        System.out.printf("Critical value (df=5): %.3f%n%n", CRITICAL_VALUE);

        if (Math.abs(tStat) > CRITICAL_VALUE) {
            System.out.println("Result: SIGNIFICANT difference (p < 0.05)");
            System.out.println(meanDiff < 0
                ? "=> Random Forest is significantly BETTER than Naive Bayes on Macro F1."
                : "=> Naive Bayes is significantly BETTER than Random Forest on Macro F1.");
        } else {
            System.out.println("Result: NO significant difference at p < 0.05");
            System.out.println("=> Cannot conclude one classifier is superior.");
        }
        System.out.printf("%nMean difference (NB - RF): %.4f  (%s tends to score higher)%n",
                meanDiff, meanDiff < 0 ? "Random Forest" : "Naive Bayes");
    }

    // ---------------------------------------------------------------
    // 5x2cv t-statistic (Dietterich, 1998, equation 3):
    //
    //   t = p_1(1) / sqrt( (1/5) * sum_i[ s_i^2 ] )
    //
    // where:
    //   s_i^2 = (p_i(1) - mean_i)^2 + (p_i(2) - mean_i)^2
    //   mean_i = (p_i(1) + p_i(2)) / 2
    // ---------------------------------------------------------------
    private static double computeTStatistic(double[] d1, double[] d2) {
        double varianceSum = 0;
        for (int i = 0; i < NUM_REPS; i++) {
            double mean = (d1[i] + d2[i]) / 2.0;
            double si2  = Math.pow(d1[i] - mean, 2) + Math.pow(d2[i] - mean, 2);
            varianceSum += si2;
        }
        double denominator = Math.sqrt(varianceSum / NUM_REPS);
        return d1[0] / denominator; // numerator = p_1(1) per Dietterich (1998)
    }

    private static double overallMeanDiff(double[] d1, double[] d2) {
        double sum = 0;
        for (int i = 0; i < NUM_REPS; i++) sum += d1[i] + d2[i];
        return sum / (NUM_REPS * 2.0);
    }

    // ---------------------------------------------------------------
    // Train on trainSet, evaluate macro F1 on testSet
    // ---------------------------------------------------------------
    private static double evalHoldout(Classifier clf, Instances trainSet, Instances testSet)
            throws Exception {
        clf.buildClassifier(trainSet);
        Evaluation eval = new Evaluation(trainSet);
        eval.evaluateModel(clf, testSet);
        return macroF1(eval, trainSet);
    }

    private static double macroF1(Evaluation eval, Instances data) throws Exception {
        double sum = 0;
        for (int i = 0; i < data.numClasses(); i++) sum += eval.fMeasure(i);
        return sum / data.numClasses();
    }

    // ---------------------------------------------------------------
    // FilteredClassifier pipeline — TF-IDF fit on training data only
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

    private static RandomForest buildRandomForest() {
        RandomForest rf = new RandomForest();
        rf.setNumIterations(RF_TREES);
        rf.setNumFeatures(RF_FEATURES);
        rf.setMaxDepth(0);
        rf.setSeed(42);
        return rf;
    }

    private static Instances loadCSV(String path) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(path));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }
}
