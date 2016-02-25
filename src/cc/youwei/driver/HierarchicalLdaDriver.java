package cc.youwei.driver;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.regex.Pattern;

import cc.mallet.pipe.CharSequence2TokenSequence;
import cc.mallet.pipe.CharSequenceLowercase;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.pipe.TokenSequenceRemoveStopwords;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.topics.HierarchicalLDA;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.InstanceList;
import cc.mallet.util.Randoms;

public class HierarchicalLdaDriver {

	public static void main(String[] args) throws IOException {
		double alpha, gamma, eta;
		String input_train, input_test, predict_test_result_dir;
		int numLevels, numberOfIterations, burnin, sampleSpace;
		
		numLevels = Integer.valueOf(args[0]);
		input_train = args[1];
		numberOfIterations = Integer.valueOf(args[2]);
		alpha = Double.valueOf(args[3]);
		gamma = Double.valueOf(args[4]);
		eta = Double.valueOf(args[5]);
		
		input_test = args[6];
		burnin = Integer.valueOf(args[7]);
		sampleSpace = Integer.valueOf(args[8]);
		predict_test_result_dir = args[9];

		InstanceList instances_train, instances_test;
		Reader fileReader_train, fileReader_test;
		
		// Begin by importing documents from text to feature sequences
		ArrayList<Pipe> pipeList_train = new ArrayList<Pipe>();

		// Pipes: tokenize, map to features
		pipeList_train.add( new CharSequence2TokenSequence(Pattern.compile("\\S+")) );
		pipeList_train.add( new TokenSequence2FeatureSequence() );
		instances_train = new InstanceList (new SerialPipes(pipeList_train));
		fileReader_train = new InputStreamReader(new FileInputStream(input_train), "UTF-8");
		instances_train.addThruPipe(new CsvIterator(fileReader_train, Pattern.compile("^(.*)$"), 1, -1, -1)); 
		                                                                              // data, label, fields
		
		// Begin by importing documents from text to feature sequences
		ArrayList<Pipe> pipeList_test = new ArrayList<Pipe>();
		// Pipes: tokenize, map to features
		pipeList_test.add( new CharSequence2TokenSequence(Pattern.compile("\\S+")) );
		pipeList_test.add( new TokenSequence2FeatureSequence() );
		instances_test = new InstanceList (new SerialPipes(pipeList_test));
		fileReader_test = new InputStreamReader(new FileInputStream(input_test), "UTF-8");
		instances_test.addThruPipe(new CsvIterator(fileReader_test, Pattern.compile("^(.*)$"), 1, -1, -1)); 
		                                                                              // data, label, fields
		HierarchicalLDA sampler = new HierarchicalLDA();
		sampler.setAlpha(alpha);
		sampler.setEta(eta);
		sampler.setGamma(gamma);
		
		sampler.initialize(instances_train, instances_test, numLevels, new Randoms(0));
		
		sampler.estimate(numberOfIterations);
		
		double[] testDocLoglik;
		double ll=0.0;
		int numberOfWords=0;
		for(int i=0; i<instances_test.size(); i++) {
			System.out.print("Predict test doc " + i + ":");
			testDocLoglik = sampler.predict(i, burnin, sampleSpace);
			ll += HierarchicalLDA.computeHarmonicMean(testDocLoglik);
			assert !Double.isNaN(ll);
			numberOfWords += ((FeatureSequence)instances_test.get(i).getData()).getLength();
			System.out.println(ll + "  " + numberOfWords);
		}
		
		File resultDir = new File(predict_test_result_dir);
		if(!resultDir.exists()) resultDir.mkdirs();
		
		File testLikelihoodFile = new File(predict_test_result_dir, "likelihood");
		File numberOfWordsFile = new File(predict_test_result_dir, "numwords");
		File perplexityFile = new File(predict_test_result_dir, "perplexity");
		
		FileWriter writer;
		
		writer = new FileWriter(testLikelihoodFile);
		writer.write(ll+"");
		writer.flush();
		writer.close();

		writer = new FileWriter(numberOfWordsFile);
		writer.write(numberOfWords+"");
		writer.flush();
		writer.close();
		
		writer = new FileWriter(perplexityFile);
		writer.write(Math.exp(-ll/numberOfWords)+"");
		writer.flush();
		writer.close();
	}

}
