package cs475;

import java.io.Serializable;
import java.util.List;

public abstract class Predictor implements Serializable {
	private static final long serialVersionUID = 1L;

	public abstract void train(List<Instance> instances, List<Instance> instances_test);
	
	public abstract void test(List<Instance> instances);
	
	public abstract Label predict(Instance instance);
}
