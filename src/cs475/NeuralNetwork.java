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
	List<double[]> totalActValues; //activation in each layers
	
	public NeuralNetwork() {
		this.totalWeights = new ArrayList<double[][]>();	
		for(int n=0; n<this.neuronNum.length-1; n++) {
			int preNeuronNum = neuronNum[n]+1; //neuron Number (In previous neuron Layer) plus one bias
			int postNeuronNum = neuronNum[n+1]; //neuron Number In next neuron Layer
//			double weights[][] = new double[preNeuronNum][postNeuronNum]; 
			
			double[][] weights = Matrix.random(preNeuronNum, postNeuronNum);
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
		
		
		
		
		return gradients;
	}

}
