package cs475;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import PrincetonMatrix.*;

public class NeuralNetwork extends Predictor{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private List<double[][]> totalWeights;
	int neuronNum[]; //neuron Number In Each Layers: 
	List<double[]> totalActValues; //activation in each layers, the first layer should be image size
	List<double[]> totalSumValues; //sum in each layers, the first layer doesn't have sum value
//	List<double[][]> gradients;
	double[] labelValue;
	long lastTime;
	
	public NeuralNetwork() {
		
		neuronNum = new int[4];
		neuronNum[0] = 1024;
		neuronNum[1] = 32;
		neuronNum[2] = 16;
		neuronNum[3] = 10;
		
		/*weight*/
		this.totalWeights = new ArrayList<double[][]>();	
		for(int n=0; n<this.neuronNum.length-1; n++) {
			int preNeuronNum = neuronNum[n]+1; //neuron Number (In previous neuron Layer) plus one bias
			int postNeuronNum = neuronNum[n+1]; //neuron Number In next neuron Layer
//			double weights[][] = new double[preNeuronNum][postNeuronNum]; 
			
			
			double[][] weights = Matrix.random(postNeuronNum, preNeuronNum);
//			double[][] weights = Matrix.ones(postNeuronNum, preNeuronNum);
			
			this.totalWeights.add(weights);
		}
		
		/*sum*/
		this.totalSumValues = new ArrayList<double[]>();
		for(int n=1; n<this.neuronNum.length; n++) {
			int neuronNum = this.neuronNum[n]; //neuron Number start from second layer
			double[] sumValues = new double[neuronNum];
			this.totalSumValues.add(sumValues);
		}
		
		/*activation*/
		this.totalActValues = new ArrayList<double[]>();
		for(int n=0; n<this.neuronNum.length; n++) {
			int neuronNum = this.neuronNum[n]+1; //neuron Number (In previous neuron Layer) plus one bias
			
			if(n==this.neuronNum.length-1) { //last layer (output layer) don't need bias
				neuronNum = neuronNum -1;
				double[] actValues = new double[neuronNum];
				this.totalActValues.add(actValues);
		  	} else {
				double[] actValues = new double[neuronNum];
				actValues[0] = 1; //bias equals to 1
				this.totalActValues.add(actValues);
			}
			
		}
		/*label*/
		int labelNum = this.neuronNum[this.neuronNum.length-1];
		this.labelValue = new double[labelNum];
		
//		this.gradients = new ArrayList<double[][]>();
//		for(int n=0; n<this.neuronNum.length-1; n++) {
//			int preNeuronNum = neuronNum[n]+1; //neuron Number (In previous neuron Layer) plus one bias
//			int postNeuronNum = neuronNum[n+1]; //neuron Number In next neuron Layer
//			double gradient[][] = new double[postNeuronNum][preNeuronNum]; 
//			gradients.add(gradient);
//		}
	}
	
	@Override
	public void train(List<Instance> instances, List<Instance> instances_test) {
		// TODO Auto-generated method stub

		sgd(instances, instances_test, 10000, 10, 3.0 );


	}

	@Override
	public Label predict(Instance instance) {
		// TODO Auto-generated method stub
		return null;
	}

	public int evaluate(Instance testData, int label) {
		int ry = 0;
		
		for(int n=0; n<testData._feature_vector.features.size(); n++) {
			double value = testData._feature_vector.features.get(n+1);
			this.totalActValues.get(0)[n+1] = value;  
		}
		
		/*feed forward*/
		for(int l=0; l<neuronNum.length-1; l++) {
			double weights[][] = this.totalWeights.get(l);
			double sumValue[] = Matrix.multiply(weights, this.totalActValues.get(l));
			
			if (neuronNum[l+1] != sumValue.length) 
				throw new RuntimeException("Illegal matrix dimensions.");
			
			for(int i=0; i<sumValue.length; i++) {
				this.totalSumValues.get(l)[i] = sumValue[i];
				if(l == neuronNum.length-2) //last year not have the bias neuron
					this.totalActValues.get(l+1)[i] = sigmo(sumValue[i]);
				else
					this.totalActValues.get(l+1)[i+1] = sigmo(sumValue[i]);
			}
			
		}
		
		double[] x = totalActValues.get(totalActValues.size()-1);
		
		double largest = Double.MIN_VALUE;
	
		for(int i =0;i<x.length;i++) {
			if(x[i] > largest) {
				largest = x[i];
				ry = i;
			}
		}
		
//		System.out.println("real:"+label+" predict:"+ry);
		
		return ry==label ? 1:0;
	}

	@Override
	public void test(List<Instance> instances) {
		
		int count = 0;
		int correct = 0;

		for (Instance instance: instances) {
			
			String label = ((ClassificationLabel)instance._label).toString();
			int labelIndex = Integer.parseInt(label);
			
			count ++;
			
			if(evaluate(instance, labelIndex) == 1) {
				correct ++;
			}
		}
		
		System.out.println("correct: "+ correct+" / "+ count);
	}
	
	public void sgd(List<Instance> trainData, List<Instance> instances_test, int iterations, int batchSize, double learningRate ) {
		for (int iter=0; iter<iterations; iter++){
			long seed = System.nanoTime();
			Collections.shuffle(trainData, new Random(seed));
//			System.out.println("shuffle");
			for (int k=0; k<trainData.size(); k+=batchSize){
				

				List<Instance> batchTrainData = trainData.subList(k, k+batchSize);
				updateWeights(batchTrainData, learningRate);
				
				
			}
			
//			System.out.println("begin testing...");
			System.out.print("iter: "+iter);
			test(instances_test);
		}
	}
	
	public void updateWeights(List<Instance> batchTrainData, double learningRate) {
		List<double[][]> gradients = new ArrayList<double[][]>();
		for(int n=0; n<this.neuronNum.length-1; n++) {
			int preNeuronNum = neuronNum[n]+1; //neuron Number (In previous neuron Layer) plus one bias
			int postNeuronNum = neuronNum[n+1]; //neuron Number In next neuron Layer
			double gradient[][] = new double[postNeuronNum][preNeuronNum]; 
			gradients.add(gradient);
		}
		
		
		List<double[][]> miniGradients = new ArrayList<double[][]>();
		for(int n=0; n<this.neuronNum.length-1; n++) {
			
			int preNeuronNum = neuronNum[n]+1; //neuron Number (In previous neuron Layer) plus one bias
			int postNeuronNum = neuronNum[n+1]; //neuron Number In next neuron Layer
			
			
			double gradient[][] = new double[postNeuronNum][preNeuronNum]; 
			
			
			miniGradients.add(gradient);
		}
		
		
		
		
		for (Instance ins: batchTrainData) {
			/*set activation value in first layer & feed forward*/
//			System.out.println("feedForward");
			
			
<<<<<<< HEAD
//			System.out.print("mini batch No:"+k+" ");
			
			feedForward(ins);
//			if(System.currentTimeMillis()-this.lastTime > 5)
//				System.out.println("feedforward time:"+ (System.currentTimeMillis()-this.lastTime)+"ms");
				
=======
>>>>>>> 60afc31481c3cf0b11defb3c3354c7c568d9008b
			/*get label value*/
			getLabelValue(ins);
			
//			System.out.println("gradient");
<<<<<<< HEAD
			List<double[][]> miniGradients = backForward();
//			if(System.currentTimeMillis()-this.lastTime > 5)
//				System.out.println("backforward time:"+ (System.currentTimeMillis()-this.lastTime)+"ms");
=======
			
			
			
			backForward(miniGradients);
			
 
>>>>>>> 60afc31481c3cf0b11defb3c3354c7c568d9008b
			
			int layerNum = miniGradients.size();
			
			
			for (int b=0; b<layerNum; b++) {
				int iMax = miniGradients.get(b).length;
				int jMax = miniGradients.get(b)[0].length;
				for(int i=0; i<iMax; i++){
					for (int j=0; j<jMax; j++){
						gradients.get(b)[i][j] += miniGradients.get(b)[i][j];
					}
				}
			}
//			this.lastTime = System.currentTimeMillis();
//			System.gc();
//			if(System.currentTimeMillis()-this.lastTime > 5)
//				System.out.println("gc time:"+ (System.currentTimeMillis()-this.lastTime)+"ms");
		}	
		
		int layerNum = gradients.size();
	
		double temp = learningRate / batchTrainData.size();
		for (int b=0; b<layerNum; b++) {
			int iMax = gradients.get(b).length;
			int jMax = gradients.get(b)[0].length;
			for(int i=0; i<iMax; i++){
				for (int j=0; j<jMax; j++){
					totalWeights.get(b)[i][j] = totalWeights.get(b)[i][j] - temp*gradients.get(b)[i][j];
				}
			}
		}
	}
	

<<<<<<< HEAD
	public List<double[][]> backForward() {
		
		this.lastTime = System.currentTimeMillis();
		
		
		List<double[][]> gradients = new ArrayList<double[][]>();
		for(int n=0; n<this.neuronNum.length-1; n++) {
			int preNeuronNum = neuronNum[n]+1; //neuron Number (In previous neuron Layer) plus one bias
			int postNeuronNum = neuronNum[n+1]; //neuron Number In next neuron Layer
			double gradient[][] = new double[postNeuronNum][preNeuronNum]; 
			
			gradients.add(gradient);
		}/**/
		/*
		for(int n=0; n<this.neuronNum.length-1; n++) {
			int preNeuronNum = neuronNum[n]+1; //neuron Number (In previous neuron Layer) plus one bias
			int postNeuronNum = neuronNum[n+1]; //neuron Number In next neuron Layer
			double gradient[][] = new double[postNeuronNum][preNeuronNum]; 
			gradients.set(n, gradient);
		}*/
		
		if(System.currentTimeMillis()-this.lastTime > 5) {
			System.out.println("assign value:"+ (System.currentTimeMillis()-this.lastTime)+"ms");
//			System.out.println("last value:"+ this.lastTime+"ms");
		}
		this.lastTime = System.currentTimeMillis();
		
=======
	public List<double[][]> backForward(List<double[][]> gradients) {

>>>>>>> 60afc31481c3cf0b11defb3c3354c7c568d9008b
		//backward 
		
		
		//1. get target error 
		int numLastLayer = neuronNum[neuronNum.length-1];
		double [] delta = new double [numLastLayer];
		
		double[] lastAct = totalActValues.get(totalActValues.size() -1);
//		StdArrayIO.print(lastAct);
		double[] s = new double[numLastLayer]; // s is the last vector calculated by the sigmoid prime
		double[] lastSum = totalSumValues.get(totalSumValues.size() - 1);
		for (int i=0; i<numLastLayer; i++){
			s[i] = sigmoPrime(lastSum[i]);
		}		
		
		// the last layer
		delta =  Matrix.multiply(Matrix.subtract(lastAct, labelValue), s) ;
		
		gradients.set(neuronNum.length-2, Matrix.multiplyTwo(delta, totalActValues.get(totalActValues.size()-2)));
		
<<<<<<< HEAD
		if(System.currentTimeMillis()-this.lastTime > 5) {
			System.out.println("last layer:"+ (System.currentTimeMillis()-this.lastTime)+"ms");
//			System.out.println("last value:"+ this.lastTime+"ms");
		}
		
		
		// the other layers
		for (int lr=neuronNum.length-3; lr>=0 ; lr--) {
			this.lastTime = System.currentTimeMillis();
=======
//		

		// the other layers
		for (int lr=neuronNum.length-3; lr>=0 ; lr--) {
			
			long starttime = System.currentTimeMillis();
>>>>>>> 60afc31481c3cf0b11defb3c3354c7c568d9008b
			
			double[] sum = totalSumValues.get(lr);
			int numNodesOfLayer = neuronNum[lr+1];
			s = new double[numNodesOfLayer];
			for(int i=0; i< numNodesOfLayer; i++) {
				s[i] = sigmoPrime(sum[i]);
			}
			if(System.currentTimeMillis()-this.lastTime > 5) {
				System.out.println("hidden layer"+lr+"(sigmo)"+":"+ (System.currentTimeMillis()-this.lastTime)+"ms");
			}
			
			// get the weight without bias
			double[][] w = getMatrixWithoutBias(totalWeights.get(lr+1));
			
			this.lastTime = System.currentTimeMillis();
			delta = Matrix.multiply(Matrix.multiply( Matrix.transpose(w), delta) , s); 
<<<<<<< HEAD
			if(System.currentTimeMillis()-this.lastTime > 5) {
				System.out.println("hidden layer"+lr+"(delta)"+":"+ (System.currentTimeMillis()-this.lastTime)+"ms");
			}
			
			this.lastTime = System.currentTimeMillis();

			gradients.set(lr, Matrix.multiplyTwo(delta, totalActValues.get(lr)));
			if(System.currentTimeMillis()-this.lastTime > 100) {
				System.out.println("hidden layer"+lr+"(gradients)"+":"+ (System.currentTimeMillis()-this.lastTime)+"ms");
			}
=======
//			gradients.set(lr, );
			Matrix.multiplyTwo(delta, totalActValues.get(lr), gradients.get(lr));
			
			
			long endtime = System.currentTimeMillis();
>>>>>>> 60afc31481c3cf0b11defb3c3354c7c568d9008b
			
			if ((endtime - starttime)>1) {
				System.out.println("back pro takes "+ (endtime - starttime));
			}
//			System.out.println("shape " + gradients.get(lr).length + " " +  gradients.get(lr)[0].length);
//			System.out.println("shape " + totalActValues.get(lr ).length + " " +totalActValues.get(lr)[0] );
		}
		

		
		
		return gradients;
	}
	
	public double[][] getMatrixWithoutBias(double[][] w) {
		double[][] result = new double[w.length][w[0].length - 1];
		
		for (int i=0; i<w.length; i++){
			for(int j=1; j<w[0].length; j++) {
				result[i][j-1] = w[i][j];
			}
		}
		return result;
	}

	public void init() { // init sum/act to zero
		
		this.totalSumValues = new ArrayList<double[]>();
		for(int n=1; n<this.neuronNum.length; n++) {
			int neuronNum = this.neuronNum[n]; //neuron Number start from second layer
			double[] sumValues = new double[neuronNum];
			this.totalActValues.set(n, sumValues);
		}
		
		this.totalActValues = new ArrayList<double[]>();
		for(int n=0; n<this.neuronNum.length; n++) {
			int neuronNum = this.neuronNum[n]+1; //neuron Number (In previous neuron Layer) plus one bias
			
			if(n==this.neuronNum.length-1) { //last layer (output layer) don't need bias
				neuronNum = neuronNum -1;
				double[] actValues = new double[neuronNum];
				this.totalActValues.set(n, actValues);
			
		  	} else {
				double[] actValues = new double[neuronNum];
				actValues[0] = 1; //bias equals to 1
				this.totalActValues.set(n, actValues);
			}	
		}
	}
	
	public void feedForward(Instance instance) {
		/*first layer of act*/
		if (neuronNum[0] != instance._feature_vector.features.size()) 
			throw new RuntimeException("Illegal matrix dimensions: number of activations in first layer doesn't match");
		
		for(int n=0; n<instance._feature_vector.features.size(); n++) {
			double value = instance._feature_vector.features.get(n+1);
			this.totalActValues.get(0)[n+1] = value;  
		}
		
		
		/*feed forward*/
		for(int l=0; l<neuronNum.length-1; l++) {
			double weights[][] = this.totalWeights.get(l);
			double sumValue[] = Matrix.multiply(weights, this.totalActValues.get(l));
			
			if (neuronNum[l+1] != sumValue.length) 
				throw new RuntimeException("Illegal matrix dimensions.");
			
			for(int i=0; i<sumValue.length; i++) {
				this.totalSumValues.get(l)[i] = sumValue[i];
				if(l == neuronNum.length-2) //last layer does not have the bias neuron
					this.totalActValues.get(l+1)[i] = sigmo(sumValue[i]);
				else
					this.totalActValues.get(l+1)[i+1] = sigmo(sumValue[i]);

			}	

		}
	}

	public void getLabelValue(Instance instance) {
		
		String label = ((ClassificationLabel)instance._label).toString();
		int labelIndex = Integer.parseInt(label);
	
		for(int n=0; n<this.labelValue.length; n++) {
			this.labelValue[n] = 0;
		}
		this.labelValue[labelIndex] = 1;
	}
	

	public double sigmo(double z) {
		return 1.0/(1+Math.exp(-z));
	}
	
	public double sigmoPrime(double z) {
		return sigmo(z)*(1-sigmo(z));
	}
}
