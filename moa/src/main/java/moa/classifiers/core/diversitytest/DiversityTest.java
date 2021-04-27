package moa.classifiers.core.diversitytest;

import com.yahoo.labs.samoa.instances.Instance;
import java.util.List;
import java.util.concurrent.Callable;

import moa.classifiers.Classifier;
import moa.options.OptionHandler;

public interface DiversityTest extends OptionHandler, Callable<Double> {
	public void set(List<Instance> testChunk, List<Classifier> targetPool);
	public boolean morePositiveMoreDiverse();
}
