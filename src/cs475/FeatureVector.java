package cs475;

import java.io.Serializable;
import java.util.HashMap;

public class FeatureVector implements Serializable {

	HashMap<Integer, Double> features = new HashMap<Integer, Double>();
	
	public void add(int index, double value) {
		// TODO Auto-generated method stub
		features.put(index,value);
	}
	
	public double get(int index) {
		// TODO Auto-generated method stub
		
		return features.get(index);
	}

}
