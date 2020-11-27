package moa.classifiers.core.diversitytest;

import java.util.ArrayList;
import java.util.List;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.Classifier;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.tasks.TaskMonitor;

public class QStatistics extends AbstractOptionHandler implements DiversityTest {

	private static final long serialVersionUID = 1L;
	
	private List<Instance> testChunk;
	private List<Classifier> classifierPool;
	
	private boolean isSet;
	
	public static Double getQScoreForTwo(List<Instance> chunk, Classifier d1, Classifier d2) {
		double tt = 0.0, tf = 0.0, ft = 0.0, ff = 0.0;
		for (Instance instance : chunk) {
			if (d1.correctlyClassifies(instance) && d2.correctlyClassifies(instance)) {
				++tt;
			} else if (d1.correctlyClassifies(instance) && !d2.correctlyClassifies(instance)) {
				++tf;
			} else if (!d1.correctlyClassifies(instance) && d2.correctlyClassifies(instance)) {
				++ft;
			} else if (!d1.correctlyClassifies(instance) && !d2.correctlyClassifies(instance)) {
				++ff;
			}
		}
		double a = tt * ff;
		double b = ft * tf;
		
		return (a - b) / (a + b);
	 }
	
	public QStatistics() {
		this.testChunk = null;
		this.classifierPool = null;
		this.isSet = false;
	}
	
	public QStatistics(List<Instance> chunk, List<Classifier> pool) {
		this.testChunk = new ArrayList<Instance>(chunk);
		this.classifierPool = new ArrayList<Classifier>(pool);
		this.isSet = true;
	}

	@Override
	public void getDescription(StringBuilder sb, int indent) {
		// TODO Auto-generated method stub

	}

	@Override
	public Double call() throws Exception {
//		return getQScore();
		if (isSet) {
			return getAverageQScore();
		} else {
			return 0.0;
		}
	}
	
	protected Double getAverageQScore() {
		double qScoreSum = 0.0;
		for (int i = 0; i < this.classifierPool.size()-1; ++i) {
			qScoreSum += QStatistics.getQScoreForTwo(this.testChunk, this.classifierPool.get(i), this.classifierPool.get(i+1));
		}
		double poolSize = this.classifierPool.size();
		return (2.0 / (poolSize*(poolSize-1.0)))*qScoreSum;
	}

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		// TODO Auto-generated method stub

	}

	@Override
	public void set(List<Instance> testChunk, List<Classifier> targetPool) {
		this.testChunk = new ArrayList<Instance>(testChunk);
		this.classifierPool = new ArrayList<Classifier>(targetPool);
		this.isSet = true;
	}

	@Override
	public boolean morePositiveMoreDiverse() {
		return false;
	}

}
