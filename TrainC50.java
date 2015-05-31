import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.StringReader;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.tokenizers.WordTokenizer;
import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.MakeIndicator;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.filters.unsupervised.instance.Resample;
import weka.filters.unsupervised.instance.SubsetByExpression;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.InfoGainAttributeEval;

/*	This is a project for Machine Learning course, to learn and predict authorship of news articles 
 *  Data source: http://archive.ics.uci.edu/ml/datasets/Reuter_50_50 
 *  ML tools: WEKA 3.7
 */
public class TrainC50 {

	String trainFileDir = "D:\\ML_HW\\project\\C50\\";
	String trainFileName = "C50_all_substrings_lowercase_5grams_train.arff";
	String testFileDir = trainFileDir;
	String testFileName = "C50_all_substrings_lowercase_5grams_test.arff";
	String outFileDir = trainFileDir;
	String outPredictFileName = "C50_result_predict_fold_";
	String outActualFileName = "C50_result_actual_fold_";
	String outUnknownClassFile = "C50_unknownClass.txt";
	boolean append = false;				// True: append to the end of current file; False: replace current file
	String classIdx = "last"; 			// The position index of class attribute
	int exp = 1; 						// Set exponent in SMO
	int targetSizePerAuthor = 25; 		// Number of positive labels in training set for each author
	int otherSizePerAuthor = 25; 		// Number of negative labels in training set for each author
	int totalAuthorNo = 50;
	int targetAuthorNo = 49;
	int folds = 5; 						// Set number of Cross Validation
	int seed = 1; 						// Used in random sampling
	int wordToKeep = 10000;
	int numToSelect = 2500;
	List<Integer> unknownClass = new ArrayList<Integer>();
	String[] resultType = { "WeightedFMeasure" };
	String[][] evalResult = new String[resultType.length][totalAuthorNo + 1];
	double[] tmpColTotal = new double[resultType.length];
	double[][] tmpRowTotal = new double[resultType.length][totalAuthorNo];
	int[] rowCount = new int[totalAuthorNo];
	int authorsNoOtherClass = 48;

	/* testMode:
	 * 1: cross validation, with all training and test samples
	 * 2: vary the positive/negative training sample size
	 * 3: vary the number of negative authors
	 * 4: final test, using the best result of 2 and 3 
	*/
	static int testMode = 4;

	public static void main(String args[]) throws Exception {

		if (testMode == 1) {
			TrainC50 c50 = new TrainC50();
			c50.initialEvalResult();
			c50.checkDirectory(c50.outFileDir, "");
			System.out.println(c50.outFileDir);

			c50.runCV(c50.trainFileDir + c50.trainFileName);
			c50.printUnknownClass(c50.outFileDir + c50.outUnknownClassFile);
			c50.printEvalResult();
		} else if (testMode == 2) {
			int[] targetInstancesSize = {1, 5, 10, 25, 40};
			int[] otherInstancesSize = {5, 10, 25, 50, 100};
			for (int i = 0; i < targetInstancesSize.length; i++) {
				for (int j = 0; j < otherInstancesSize.length; j++) {

					TrainC50 c50 = new TrainC50();
					c50.initialEvalResult();
					//c50.authorsNoOtherClass = 5;
					c50.targetSizePerAuthor = targetInstancesSize[i];
					c50.otherSizePerAuthor = otherInstancesSize[j];
					String newDir = "T" + targetInstancesSize[i] + "_O"
							+ otherInstancesSize[j] + "\\";
					c50.checkDirectory(c50.outFileDir, newDir);
					c50.outFileDir = c50.outFileDir + newDir;
					c50.runCV(c50.trainFileDir + c50.trainFileName);
					c50.printUnknownClass(c50.outFileDir
							+ c50.outUnknownClassFile);
					c50.printEvalResult();
				}
			}
		} else if (testMode == 3) {
			int[] authorsNoOtherClass = { 1, 2, 5, 10 };
			for (int i = 0; i < authorsNoOtherClass.length; i++) {
				TrainC50 c50 = new TrainC50();
				c50.initialEvalResult();
				c50.authorsNoOtherClass = authorsNoOtherClass[i];
				String newDir = "OAuthorsNo_" + authorsNoOtherClass[i] + "/";
				c50.checkDirectory(c50.outFileDir, newDir);
				c50.outFileDir = c50.outFileDir + newDir;
				c50.runCV(c50.trainFileDir + c50.trainFileName);
				c50.printUnknownClass(c50.outFileDir + c50.outUnknownClassFile);
				c50.printEvalResult();
			}
		} else {
			TrainC50 c50 = new TrainC50();
			c50.initialEvalResult();
			c50.targetSizePerAuthor = 10;
			c50.otherSizePerAuthor = 10;
			c50.authorsNoOtherClass = 5;
			String newDir = "";
			c50.checkDirectory(c50.outFileDir, newDir);
			c50.outFileDir = c50.outFileDir + newDir;
			c50.runTest(c50.trainFileDir + c50.trainFileName, c50.testFileDir
					+ c50.testFileName);
			c50.printUnknownClass(c50.outFileDir + c50.outUnknownClassFile);
			c50.printEvalResult();
		}
	}

	public Instances getDataInstances(String filePath) throws Exception {
		DataSource source = new DataSource(filePath);
		Instances data = source.getDataSet();
		// Set the position of class label
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		return (data);
	}

	// Used for cross validation
	public void runCV(String trainFilePath) throws Exception {

		// Load data from file
		Instances wholedata = getDataInstances(trainFilePath);

		// stratify the data for CV
		if (wholedata.classAttribute().isNominal()) {
			wholedata.stratify(folds);
		}
		// Separate training and testing set to do CV
		for (int n = 0; n < folds; n++) {
			Instances train = wholedata.trainCV(folds, n);
			Instances test = wholedata.testCV(folds, n);

			String outPredictFile = outFileDir + outPredictFileName + n + ".txt";
			String outActualFile = outFileDir + outActualFileName + n + ".txt";
			printResult(outPredictFile, outActualFile, runTraining(train, test));
		}
	}

	// Used for the final test
	public void runTest(String trainFilePath, String testFilePath)
			throws Exception {

		// Load train and test data from file
		Instances train = getDataInstances(trainFilePath);
		Instances test = getDataInstances(testFilePath);

		String outPredictFile = outFileDir + outPredictFileName + ".txt";
		String outActualFile = outFileDir + outActualFileName + ".txt";
		printResult(outPredictFile, outActualFile, runTraining(train, test));
	}

	public List<List<Prediction>> runTraining(Instances train, Instances test)
			throws Exception {

		// Get a random number between 1 and totalAuthorNo, and assign the
		// author to be unknown class
		Random r = new Random(getNextSeed());
		int x = r.nextInt(totalAuthorNo);
		unknownClass.add(x);

		// Remove the unknown author from training set
		Instances newTrain = new Instances(train);
		newTrain = removeClass(newTrain, x);

		// Merge unknown texts from train and test set to get bigger test set
		//Instances unknownTrain = removeInverseClass(train, x);
		Instances unknownTest = removeInverseClass(test, x);
		
		List<List<Prediction>> predictions = new ArrayList<List<Prediction>>();

		// for each author, use this author as unknown class and other authors as known class to train the model
		for (int i = 1; i < totalAuthorNo + 1; i++) {

			// If it's the unknown class, then skip training
			if (x == i) {
				storeEvalResult(null, i - 1);
				continue;
			}

			// Subset target author
			Instances tmpT = removeInverseClass(newTrain, i);
			tmpT = getResample(tmpT,
					(double) targetSizePerAuthor / tmpT.numInstances() * 100);
			// Subset outlier authors
			Instances tmpO = removeClass(newTrain, i);
			tmpO = removeNonTrainingAuthor(tmpO, x, i);
			tmpO = getResample(tmpO,
					(double) otherSizePerAuthor / tmpO.numInstances() * 100);

			// Merge two datasets, relabel to 1s and 0s, Set class type to be
			// nominal
			Instances trainFinal = mergeInstances(tmpT, tmpO);
			trainFinal = relabel(trainFinal, i);
			trainFinal = setNominalClass(trainFinal, classIdx);

			// Apply the same filter (Relabel, set Nominal) to testset
			// Only use target and unknown author texts in test set
			Instances testFinal = new Instances(test);
			testFinal = removeInverseClass(testFinal, i);
			testFinal = mergeInstances(testFinal, unknownTest);
			testFinal = relabel(testFinal, i);
			testFinal = setNominalClass(testFinal, classIdx);

			// Run classifier with multiple filters, and add test result into
			// predictions list to print
			// Filters
			MultiFilter multifilters = new MultiFilter();
			multifilters.setFilters(new Filter[] { getWordVector(wordToKeep),
					selectAttributes() });

			// Classifier
			SMO smo = new SMO(); // base classifier
			PolyKernel poly = new PolyKernel();// Use PolyKernel
			poly.setExponent(exp);
			smo.setKernel(poly);
			// make cost sensitive to upweight false positives
			Classifier finalClassifier = makeCostSensitiveClassifier(smo,
					(double) 1, (double) otherSizePerAuthor
							/ targetSizePerAuthor); 

			Evaluation eval = runFilteredClassifier(trainFinal, testFinal,
					multifilters, finalClassifier);
			
			predictions.add(eval.predictions());
			storeEvalResult(eval, i - 1);

		}

		return (predictions);
	}

	// Make cost-sensitive classifier to upweight false positives
	public CostSensitiveClassifier makeCostSensitiveClassifier(
			Classifier classifier, double falsePositive, double falseNegative)
			throws Exception {
		CostSensitiveClassifier csc = new CostSensitiveClassifier();
		csc.setClassifier(classifier);
		csc.setMinimizeExpectedCost(false);
		// upweight false positive
		String costString = "[0.0 " + falsePositive + "; " + falseNegative
				+ " 0.0]"; 
		StringWriter writer = new StringWriter();
		CostMatrix.parseMatlab(costString).write(writer);
		csc.setCostMatrix(new CostMatrix(new StringReader(writer.toString())));
		csc.setCostMatrixSource(new SelectedTag(
				CostSensitiveClassifier.MATRIX_SUPPLIED,
				CostSensitiveClassifier.TAGS_MATRIX_SOURCE));
		return (csc);
	}

	// Run SMO with filterClassifier
	public Evaluation runFilteredClassifier(Instances train, Instances test,
			MultiFilter multifilters, Classifier classifier) throws Exception {
		FilteredClassifier f = new FilteredClassifier();
		f.setClassifier(classifier);
		f.setFilter(multifilters);
		f.buildClassifier(train);

		// Start testing
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(f, test);

		return (eval);
	}

	public Instances mergeInstances(Instances data1, Instances data2)
			throws Exception {
		Instances mergedData = new Instances(data1);
		for (int j = 0; j < data2.numInstances(); j++) {
			mergedData.add(data2.get(j));
			mergedData.get(mergedData.numInstances() - 1).setValue(0,
					data2.get(j).stringValue(0));
		}
		return (mergedData);
	}

	public Instances removeClass(Instances data, int idx) throws Exception {
		RemoveWithValues f = new RemoveWithValues();
		f.setSplitPoint(0.0);
		f.setNominalIndices(String.valueOf(idx));
		f.setAttributeIndex(classIdx);
		f.setInputFormat(data);
		Instances result = Filter.useFilter(data, f);
		return (result);
	}

	public Instances removeInverseClass(Instances data, int idx)
			throws Exception {
		return (removeInverseClass(data, String.valueOf(idx)));
	}

	public Instances removeInverseClass(Instances data, String idx)
			throws Exception {
		RemoveWithValues f = new RemoveWithValues();
		f.setSplitPoint(0.0);
		f.setNominalIndices(idx);
		f.setAttributeIndex(classIdx);
		f.setInvertSelection(true);
		f.setInputFormat(data);
		Instances result = Filter.useFilter(data, f);
		return (result);
	}

	public Instances relabel(Instances data, int idx) throws Exception {
		MakeIndicator f = new MakeIndicator();
		f.setAttributeIndex(classIdx);
		f.setValueIndex(idx - 1);
		f.setInputFormat(data);
		Instances result = Filter.useFilter(data, f);
		return (result);
	}

	public Instances getSubsetByClass(Instances data, String label)
			throws Exception {
		SubsetByExpression f = new SubsetByExpression();
		f.setExpression("CLASS = " + label);
		f.setInputFormat(data);
		Instances result = Filter.useFilter(data, f);
		return (result);
	}

	public AttributeSelection selectAttributes() throws Exception {
		AttributeSelection f = new AttributeSelection();
		InfoGainAttributeEval eval = new InfoGainAttributeEval();
		Ranker search = new Ranker();
		// search.setThreshold(0.0);
		search.setNumToSelect(numToSelect);
		f.setEvaluator(eval);
		f.setSearch(search);
		return (f);
	}

	public StringToWordVector getWordVector(int nWords) throws Exception {
		StringToWordVector f = new StringToWordVector();
		f.setIDFTransform(true);
		f.setDoNotOperateOnPerClassBasis(true);
		f.setTFTransform(true);
		f.setNormalizeDocLength(new SelectedTag(
				StringToWordVector.FILTER_NORMALIZE_ALL,
				StringToWordVector.TAGS_FILTER));
		WordTokenizer wt = new WordTokenizer();
		String delimiters = ":";
		wt.setDelimiters(delimiters);
		f.setTokenizer(wt);
		f.setWordsToKeep(nWords);
		f.setMinTermFreq(2);
		f.setOutputWordCounts(true);

		return (f);
	}

	public Instances getResample(Instances data, double percent)
			throws Exception {
		Resample f = new Resample();
		f.setSampleSizePercent(percent);
		f.setNoReplacement(true);

		f.setInputFormat(data);
		Instances result = Filter.useFilter(data, f);
		return (result);
	}

	public Instances setNominalClass(Instances data, String classIdx)
			throws Exception {
		NumericToNominal f = new NumericToNominal();
		f.setAttributeIndices(classIdx);

		f.setInputFormat(data);
		Instances result = Filter.useFilter(data, f);
		return (result);
	}

	// Return a different random seed each time
	public int getNextSeed() {
		seed += 1;
		return (seed);
	}

	// Output file service
	public PrintWriter getWriter(String file, boolean append) throws Exception {
		File f = new File(file);
		if (f.exists()) {
			return (new PrintWriter(new BufferedWriter(new FileWriter(file,
					append))));
		} else {
			f.createNewFile();
			return (new PrintWriter(new BufferedWriter(
					new FileWriter(f, append))));
		}
	}

	// create folder
	public void checkDirectory(String parentDir, String newDir) {
		File dir = new File(parentDir + newDir);

		if (!dir.exists()) {
			dir.mkdir();
		}

	}

	// Output test result
	public void printResult(String outPredictFile, String outActualFile,
			List<List<Prediction>> result) throws Exception {
		PrintWriter outP = getWriter(outPredictFile, append);
		PrintWriter outA = getWriter(outActualFile, append);

		for (List<Prediction> rowOfAuthor : result) {
			String sbPredict = "";
			String sbActual = "";
			for (Prediction colOfAuthor : rowOfAuthor) {
				sbPredict += (int) colOfAuthor.predicted() + " ";
				sbActual += (int) colOfAuthor.actual() + " ";
			}
			outP.println(sbPredict);
			outA.println(sbActual);
		}
		outP.close();
		outA.close();
	}

	// store F-measure for each test result and calculate the average F-measure
	public void storeEvalResult(Evaluation eval, int authorIdx)
			throws Exception {

		// reset tmpTotal to 0
		if (authorIdx == 0) {
			for (int i = 0; i < resultType.length; i++) {
				tmpColTotal[i] = 0;
			}
		}

		// if it's unknown class, then add a empty result
		if (eval == null) {
			for (int i = 0; i < resultType.length; i++) {
				evalResult[i][authorIdx] += ",";
			}
		} else {
			double[] value = new double[resultType.length];
			value[0] = eval.weightedFMeasure();
			for (int j = 0; j < value.length; j++) {
				evalResult[j][authorIdx] += String.valueOf(value[j]) + ",";
				tmpColTotal[j] += value[j];
				tmpRowTotal[j][authorIdx] += value[j];
			}
			rowCount[authorIdx] += 1;
		}

		// if it's the end of a training, calculate average for this column
		if (authorIdx + 1 == totalAuthorNo) {
			for (int i = 0; i < resultType.length; i++) {
				evalResult[i][authorIdx + 1] += String.valueOf(tmpColTotal[i]
						/ targetAuthorNo)
						+ ",";
			}
		}

	}

	public void printEvalResult() throws Exception {

		// count for row average
		int totalCount = 0;
		double[] overallSum = new double[resultType.length];
		for (int i = 0; i < totalAuthorNo; i++) {
			for (int j = 0; j < resultType.length; j++) {
				double rowAvg = tmpRowTotal[j][i] / rowCount[i];
				evalResult[j][i] += String.valueOf(rowAvg);
				overallSum[j] += tmpRowTotal[j][i];
			}
			totalCount += rowCount[i];
		}

		// print result
		for (int i = 0; i < resultType.length; i++) {
			// add average of everything to the end of last row (avg row)
			evalResult[i][totalAuthorNo] += String.valueOf(overallSum[i]
					/ totalCount);

			PrintWriter out = getWriter(outFileDir + resultType[i] + ".csv",
					append);
			String[] content = evalResult[i];
			for (String row : content) {
				out.println(row);
			}
			out.close();
		}
	}

	// Output unknown class
	public void printUnknownClass(String outfile) throws Exception {
		PrintWriter out = getWriter(outfile, append);

		StringBuffer sb = new StringBuffer();
		for (Integer classidx : unknownClass) {
			sb.append(classidx);
			sb.append(" ");
		}
		out.println(sb.toString());
		out.close();
	}

	// Output data instances for debug / UI test
	public void printInstances(Instances data, String outfile) throws Exception {
		PrintWriter out = getWriter(outfile, append);
		out.print(data.toString());
		out.close();
	}

	public void initialEvalResult() {
		for (int i = 0; i < evalResult.length; i++) {
			String[] row = evalResult[i];
			for (int j = 0; j < row.length; j++) {
				row[j] = "";
			}
		}
	}

	public Instances removeNonTrainingAuthor(Instances data, int unknown,
			int target) throws Exception {
		// if don't need to remove class, return directly
		if (authorsNoOtherClass == (totalAuthorNo - 2))
			return (data);

		// get a random list between 1 to totalAuthorNo, avoiding unknown class
		HashSet<Integer> set = new HashSet<Integer>();
		randomSet(1, totalAuthorNo, authorsNoOtherClass, set, unknown, target);
		String perserveAuthorList = "";
		for (int j : set) {
			perserveAuthorList += "," + j;
		}
		perserveAuthorList.replaceFirst(",", "");

		return (removeInverseClass(data, perserveAuthorList));
	}

	// get a random set of n distinct numbers, between min and max, avoiding avoidMe1 and avoidMe2
	public static void randomSet(int min, int max, int n, HashSet<Integer> set,
			int avoidMe1, int avoidMe2) {
		if (n > (max - min + 1) || max < min) {
			return;
		}

		while (set.size() < n) {
			int num = (int) (Math.random() * (max - min)) + min;
			if (num != avoidMe1 && num != avoidMe2) {
				set.add(num);
			}
		}
	}
}
