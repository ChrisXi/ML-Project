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

	public void feedForward(Instance instance) {
		/*first layer of act*/
		for(int n=0; n<instance._feature_vector.features.size(); n++) {
			double value = instance._feature_vector.features.get(n);
			this.totalActValues.get(0)[n+1] = value;  
		}
		
		/*feed forward*/
		for(int l=0; l<neuronNum.length; l++) {
			double weights[][] = this.totalWeights.get(l);
			double sumValue[] = Matrix.multiply(weights, this.totalActValues.get(l));
			
			if (neuronNum[l+1] != sumValue.length) 
				throw new RuntimeException("Illegal matrix dimensions.");
			
//			for(int i=0; i<temp.length; i++) {
//				sumValue[i]
//			}
			
		}
	}
}
