package cs475;

import java.util.ArrayList;
import java.util.List;

import PrincetonMatrix.Matrix;

public class NeuralNetwork extends Predictor{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private List<double[][]> totalWeights;
	int neuronNum[]; //neuron Number In Each Layers
	List<double[]> totalActValues; //activation in each layers, the first layer should be image size
	List<double[]> totalSumValues; //sum in each layers, the first layer doesn't have sum value
	
	public NeuralNetwork() {
		this.totalWeights = new ArrayList<double[][]>();	
		for(int n=0; n<this.neuronNum.length-1; n++) {
			int preNeuronNum = neuronNum[n]+1; //neuron Number (In previous neuron Layer) plus one bias
			int postNeuronNum = neuronNum[n+1]; //neuron Number In next neuron Layer
//			double weights[][] = new double[preNeuronNum][postNeuronNum]; 
			
			double[][] weights = Matrix.random(postNeuronNum, preNeuronNum);
			this.totalWeights.add(weights);
		}
		
		this.totalActValues = new ArrayList<double[]>();
		for(int n=0; n<this.neuronNum.length-1; n++) {
			int neuronNum = this.neuronNum[n]+1; //neuron Number (In previous neuron Layer) plus one bias
			double[] actValues = new double[neuronNum];
			actValues[0] = 1; //bias equals to 1
			this.totalActValues.add(actValues);
		}
		
		this.totalSumValues = new ArrayList<double[]>();
		for(int n=1; n<this.neuronNum.length-1; n++) {
			int neuronNum = this.neuronNum[n]; //neuron Number start from second layer
			double[] sumValues = new double[neuronNum];
			this.totalActValues.add(sumValues);
		}
	}
	
	@Override
	public void train(List<Instance> instances) {
		// TODO Auto-generated method stub
		
		for (Instance instance : instances) {
			
			feedForward(instance);
			
			
		}
		
		
	}

	@Override
	public Label predict(Instance instance) {
		// TODO Auto-generated method stub
		return null;
	}
	
	
	
	public List<double[][]> backForward() {
		
		List<double[][]> gradients = new ArrayList<double[][]>();
		for(int n=0; n<this.neuronNum.length-1; n++) {
			int preNeuronNum = neuronNum[n]+1; //neuron Number (In previous neuron Layer) plus one bias
			int postNeuronNum = neuronNum[n+1]; //neuron Number In next neuron Layer
			double gradient[][] = new double[preNeuronNum][postNeuronNum]; 
			gradients.add(gradient);
		}
		
		
		//backward 
		
		//1. get target error 
		int numLastLayer = neuronNum[neuronNum.length-1];
		double [] delta = new double [numLastLayer];
		
		double[] lastAct = totalActValues.get(totalActValues.size() -1);
		double[] s = new double[numLastLayer]; // s is the last vector calculated by the sigmoid prime
		double[] lastSum = totalSumValues.get(totalSumValues.size() - 1);
		for (int i=0; i<numLastLayer; i++){
			s[i] = sigmoPrime(lastSum[i]);
		}		
		
		
		delta =  Matrix.multiply(Matrix.subtract(lastAct, y), lastSum) ;
		
		
		
		
		
		
		
		
		return gradients;
	}

	public void feedForward(Instance instance) {
		/*first layer of act*/
		for(int n=0; n<instance._feature_vector.features.size(); n++) {
			double value = instance._feature_vector.features.get(n);
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
				this.totalActValues.get(l+1)[i+1] = sigmo(sumValue[i]);
			}
			
		}
	}
	

	public double sigmo(double z) {
		return 1/1+Math.exp(-z);
	}
	
	public double sigmoPrime(double z) {
		return sigmo(z)*(1-sigmo(z));
	}
}
