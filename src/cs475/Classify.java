package cs475;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;

public class Classify {
	static public LinkedList<Option> options = new LinkedList<Option>();
	
//	-algorithm nn -data ../ML_Data/train.txt -test ../ML_Data/test.txt
	
	public static void main(String[] args) throws IOException {
		// Parse the command line.
		String[] manditory_args = {"data"};
		createCommandLineOptions();
		CommandLineUtilities.initCommandLineParameters(args, Classify.options, manditory_args);
	
//		String mode = CommandLineUtilities.getOptionValue("mode");
//		String predictions_file = CommandLineUtilities.getOptionValue("predictions_file");
//		String model_file = CommandLineUtilities.getOptionValue("model_file");
		
		String data = CommandLineUtilities.getOptionValue("data");
		String test = CommandLineUtilities.getOptionValue("test");
		String algorithm = CommandLineUtilities.getOptionValue("algorithm");
		
	
		if (data == null || algorithm == null){
			System.out.println("Train requires the following arguments: data, algorithm, model_file");
			System.exit(0);
		}
		// Load the training data.
		DataReader data_reader = new DataReader(data, true);
		List<Instance> instances = data_reader.readData();
		
		DataReader data_test_reader = new DataReader(test, true);
		List<Instance> instances_test = data_test_reader.readData();
		
		data_reader.close();
		data_test_reader.close();
		
		System.out.println("Loading data done!");
		System.out.println("Train Instance number:" + instances.size());
		System.out.println("Train Feature number: " + instances.get(0)._feature_vector.features.size());
		System.out.println("Test Instance number:" + instances_test.size());
		System.out.println("Test Feature number: " + instances_test.get(0)._feature_vector.features.size());
		// Train the model.
		Predictor predictor = train(instances, instances_test, algorithm);
		
//		evaluateAndSavePredictions(predictor, instances, predictions_file);

	}
	

	private static Predictor train(List<Instance> instances, List<Instance> instances_test, String algorithm) {
		// TODO Train the model using "algorithm" on "data"
		// TODO Evaluate the model
		
		Predictor classifier;
		if(algorithm.equals("nn")) { //neural network
			classifier = new NeuralNetwork();
			classifier.train(instances, instances_test);
//			classifier.test(instances_test);
//			evaluateAfterTrain(instances,classifier);
		} else if(algorithm.equals("cnn")) { // convolutional neural network
			
		} 
		
		return null;
	}

	private static void evaluateAndSavePredictions(Predictor predictor,
			List<Instance> instances, String predictions_file) throws IOException {
		PredictionsWriter writer = new PredictionsWriter(predictions_file);
		// TODO Evaluate the model if labels are available. 
		
//		for (Instance instance : instances) {
//			Label label = predictor.predict(instance);
//			writer.writePrediction(label);
//		}
//		
//		writer.close();
		
	}

	public static void saveObject(Object object, String file_name) {
		try {
			ObjectOutputStream oos =
				new ObjectOutputStream(new BufferedOutputStream(
						new FileOutputStream(new File(file_name))));
			oos.writeObject(object);
			oos.close();
		}
		catch (IOException e) {
			System.err.println("Exception writing file " + file_name + ": " + e);
		}
	}

	/**
	 * Load a single object from a filename. 
	 * @param file_name
	 * @return
	 */
	public static Object loadObject(String file_name) {
		ObjectInputStream ois;
		try {
			ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(new File(file_name))));
			Object object = ois.readObject();
			ois.close();
			return object;
		} catch (IOException e) {
			System.err.println("Error loading: " + file_name);
		} catch (ClassNotFoundException e) {
			System.err.println("Error loading: " + file_name);
		}
		return null;
	}
	
	public static void registerOption(String option_name, String arg_name, boolean has_arg, String description) {
		OptionBuilder.withArgName(arg_name);
		OptionBuilder.hasArg(has_arg);
		OptionBuilder.withDescription(description);
		Option option = OptionBuilder.create(option_name);
		
		Classify.options.add(option);		
	}
	
	private static void createCommandLineOptions() {
		registerOption("data", "String", true, "The train data to use.");
		registerOption("test", "String", true, "The test data to use.");
		registerOption("algorithm", "String", true, "The name of the algorithm for training.");
//		registerOption("mode", "String", true, "Operating mode: train or test.");
//		registerOption("predictions_file", "String", true, "The predictions file to create.");
//		registerOption("model_file", "String", true, "The name of the model file to create/load.");
		
		// Other options will be added here.
	}
}
