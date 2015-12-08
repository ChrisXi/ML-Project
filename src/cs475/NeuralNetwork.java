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
	int neuronNum[]; //neuron Number In Each Layers: 
	List<double[]> totalActValues; //activation in each layers, the first layer should be image size
	List<double[]> totalSumValues; //sum in each layers, the first layer doesn't have sum value
	double[] labelValue;
	
	public NeuralNetwork() {
		
		neuronNum = new int[3];
		neuronNum[0] = 1024;
		neuronNum[1] = 20;
		neuronNum[2] = 10;
		
		
		/*weight*/
		this.totalWeights = new ArrayList<double[][]>();	
		for(int n=0; n<this.neuronNum.length-1; n++) {
			int preNeuronNum = neuronNum[n]+1; //neuron Number (In previous neuron Layer) plus one bias
			int postNeuronNum = neuronNum[n+1]; //neuron Number In next neuron Layer
//			double weights[][] = new double[preNeuronNum][postNeuronNum]; 
			
			double[][] weights = Matrix.random(postNeuronNum, preNeuronNum);
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
		
	}
	
	@Override
	public void train(List<Instance> instances) {
		// TODO Auto-generated method stub
		
		for (Instance instance : instances) {
			
			/*set activation value in first layer & feed forward*/
			feedForward(instance);
			
			/*get label value*/
			getLabelValue(instance);
			
			return;
		}
		
		
	}

	@Override
	public Label predict(Instance instance) {
		// TODO Auto-generated method stub
		return null;
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
				if(i == sumValue.length-1) //last year not have the bias neuron
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
		return 1/1+Math.exp(-z);
	}
	
	public double sigmoPrime(double z) {
		return sigmo(z)*(1-sigmo(z));
	}
}
