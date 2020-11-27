/*
 *    CDCMS.java
 *    Copyright (C) 2018 University of Birmingham, Birmingham, United Kingdom
 *    @author Chun Wai Chiu (cxc1015@student.bham.ac.uk)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.classifiers.meta;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.diversitytest.QStatistics;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.core.AutoClassDiscovery;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.Utils;
import moa.options.ClassOption;

public class CDCMS extends AbstractClassifier implements MultiClassClassifier {

	/**
	 * Default serial version ID
	 */
	private static final long serialVersionUID = 1L;
	
	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "The Base Learner.", Classifier.class, "trees.HoeffdingTree -l NB"); //trees.HoeffdingTree -e 2000000 -g 100 -c 0.01
	
	public IntOption poolSizeOption = new IntOption("ensembleSize", 'k',
			"The maximum size of the ensemble.", 10, 1, Integer.MAX_VALUE);
	
	public IntOption repositorySizeOption = new IntOption("repositorySizeMultiple", 'n',
			"The repository size will be n*k", 10, 1, Integer.MAX_VALUE);
	
	public IntOption windowSizeOption = new IntOption("windowSize", 'b',
			"The window size used for classifier creation and evaluation.", 500, 1, Integer.MAX_VALUE);
	
	public FloatOption fadingFactorOption = new FloatOption("fadingFactor", 'f',
			"Fading Factor for prequential accuracy calculation on test chunk", 0.999);
	
	public FloatOption similarityThresholdOption = new FloatOption("similarityThreshold", 's',
			"similarityThreshold", 0.8, 0.0, 1.0);
	
	public ClassOption driftDetectorOption = new ClassOption("driftDetector", 'd',
            "Drift detection method to use.", ChangeDetector.class, "ADWINChangeDetector");
	
	public MultiChoiceOption wekaAlgorithmOption;

	public StringOption parameterOption = new StringOption("parameter", 'p',
            "Parameters that will be passed to the weka algorithm. (e.g. '-N 5' for using SimpleKmeans with 5 clusters)",
            "-I 100 -N -1 -X 10 -max -1 -ll-cv 1.0E-6 -ll-iter 1.0E-6 -M 1.0E-6 -K 10 -num-slots 1 -S 100");
	
	protected double similarityThreshold;
	
	protected EnsembleWithInfo ensemble_NL;
	
	protected EnsembleWithInfo ensemble_NH;
	
	protected EnsembleWithInfo ensemble_OL;
	
	protected ClassifierWithInfo candidate;
	
	protected List<ClassifierWithInfo> repository;
	protected int maxRepositorySize;
	
	protected ChangeDetector driftDetector;
	
	protected List<Instance> instWindow;
	protected int instSeenAfterDrift;
	
	private Instances predictionErrorByClassifierFromRepo;
	
	protected double warningDetected;
    protected double changeDetected;
    
    protected DRIFT_LEVEL previous_drift_level;
    protected DRIFT_LEVEL drift_level;
	
	private Class<?>[] clustererClasses;
	
	private weka.clusterers.AbstractClusterer clusterer;
	
	protected SamoaToWekaInstanceConverter instanceConverter;
	
	public CDCMS() {
		this.clustererClasses = findWekaClustererClasses();
        String[] optionLabels = new String[clustererClasses.length];
        String[] optionDescriptions = new String[clustererClasses.length];

        for (int i = 0; i < this.clustererClasses.length; i++) {
            optionLabels[i] = this.clustererClasses[i].getSimpleName();
            optionDescriptions[i] = this.clustererClasses[i].getName();
        }

        if (this.clustererClasses != null && this.clustererClasses.length > 0) {
            wekaAlgorithmOption = new MultiChoiceOption("clusterer", 'w',
                    "Weka cluster algorithm to use.",
                    optionLabels, optionDescriptions, 2);
        } else {
            parameterOption = null;

        }
	}
	
	@Override
	public boolean isRandomizable() {
		return false;
	}
	
	@Override
	public void resetLearningImpl() {
		
		// *-1, because more negative means more diverse in QStatistics.
		this.similarityThreshold = this.similarityThresholdOption.getValue() * -1;
		
		this.driftDetector = ((ChangeDetector) getPreparedClassOption(this.driftDetectorOption)).copy();
		
		this.candidate = new ClassifierWithInfo(((Classifier) this.getPreparedClassOption(this.baseLearnerOption)).copy(),
												this.fadingFactorOption.getValue());
		
		this.ensemble_NL = new EnsembleWithInfo(this.fadingFactorOption.getValue(), true, "NL");
		this.ensemble_NL.add(new ClassifierWithInfo(((Classifier) this.getPreparedClassOption(this.baseLearnerOption)).copy(),
													this.fadingFactorOption.getValue()));
		
		this.ensemble_OL = null;
		this.ensemble_NH = null;
		
		this.maxRepositorySize = this.repositorySizeOption.getValue() * this.poolSizeOption.getValue();
		this.repository = new ArrayList<ClassifierWithInfo>(this.maxRepositorySize);
		
		this.instWindow = new ArrayList<Instance>(this.windowSizeOption.getValue());
		this.instSeenAfterDrift = 0;
		
		// Setting up Instances object for clustering classifiers.
		this.initPredictionErrorStorage(this.windowSizeOption.getValue());
		//====================================================================
		this.resetClusterer();
		
		this.instanceConverter = new SamoaToWekaInstanceConverter();
		
		this.changeDetected = 0;
        this.warningDetected = 0;
        this.previous_drift_level = DRIFT_LEVEL.NORMAL;
        this.drift_level = DRIFT_LEVEL.NORMAL;
        
	}
	
	private void initPredictionErrorStorage(int numAtt) {
		Attribute[] attributes = new Attribute[numAtt + 1];
		for (int i = 0; i < attributes.length - 1; ++i) {
			attributes[i] = new Attribute("Prediction " + (i+1));
		}
		attributes[attributes.length - 1] = new Attribute("Cluster Number");
		
		this.predictionErrorByClassifierFromRepo = new Instances("predictionErrorByClassifierFromRepo",
																 attributes,
																 this.repository.size());
		this.predictionErrorByClassifierFromRepo.setClassIndex(this.predictionErrorByClassifierFromRepo.numAttributes() - 1);
	}
	
	private void resetClusterer() {
		try {
            String clistring = clustererClasses[wekaAlgorithmOption.getChosenIndex()].getName();
            this.clusterer = (weka.clusterers.AbstractClusterer) ClassOption.cliStringToObject(clistring, weka.clusterers.Clusterer.class, null);

            String rawOptions = parameterOption.getValue();
            String[] options = rawOptions.split(" ");
            if (this.clusterer instanceof weka.core.OptionHandler) {
                ((weka.core.OptionHandler) this.clusterer).setOptions(options);
                Utils.checkForRemainingOptions(options);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		
		double[] to_return = null;
		
		double accuracy_NL = this.ensemble_NL.getPrequentialAccuracy();
		double accuracy_OL = this.ensemble_OL == null ? 0.0 : this.ensemble_OL.getPrequentialAccuracy();
		double accuracy_NH = this.ensemble_NH == null ? 0.0 : this.ensemble_NH.getPrequentialAccuracy();
		
		double accuracySum = accuracy_OL + accuracy_NH + accuracy_NL;
		
		DoubleVector combinedVote = new DoubleVector();
		
		switch (this.drift_level) {
			case NORMAL:
				if (this.ensemble_OL != null && this.ensemble_NH != null &&
						accuracy_NL < accuracy_OL && accuracy_NL < accuracy_NH) {
					
					if (this.ensemble_OL.estimation > 0.0) {
						DoubleVector vote = new DoubleVector(this.ensemble_OL.getVotesForInstance(inst));
						if (vote.sumOfValues() > 0.0) {
							vote.normalize();
							vote.scaleValues(accuracy_OL / accuracySum);
							combinedVote.addValues(vote);
						}
					}
					if (this.ensemble_NH.estimation > 0.0) {
						DoubleVector vote = new DoubleVector(this.ensemble_NH.getVotesForInstance(inst));
						if (vote.sumOfValues() > 0.0) {
							vote.normalize();
							vote.scaleValues(accuracy_NH / accuracySum);
							combinedVote.addValues(vote);
						}
					}
					if (this.ensemble_NL.estimation > 0.0) {
						DoubleVector vote = new DoubleVector(this.ensemble_NL.getVotesForInstance(inst));
						if (vote.sumOfValues() > 0.0) {
							vote.normalize();
							vote.scaleValues(accuracy_NL / accuracySum);
							combinedVote.addValues(vote);
						}
					}
					to_return = combinedVote.getArrayRef();
					
				} else {
					to_return = this.ensemble_NL.getVotesForInstance(inst);
				}
				
					
				break;
			case OUTCONTROL:
				
				if (this.ensemble_OL.estimation > 0.0) {
					DoubleVector vote = new DoubleVector(this.ensemble_OL.getVotesForInstance(inst));
					if (vote.sumOfValues() > 0.0) {
						vote.normalize();
						vote.scaleValues(accuracy_OL / accuracySum);
						combinedVote.addValues(vote);
					}
				}
				if (this.ensemble_NH.estimation > 0.0) {
					DoubleVector vote = new DoubleVector(this.ensemble_NH.getVotesForInstance(inst));
					if (vote.sumOfValues() > 0.0) {
						vote.normalize();
						vote.scaleValues(accuracy_NH / accuracySum);
						combinedVote.addValues(vote);
					}
				}
				if (this.ensemble_NL.estimation > 0.0) {
					DoubleVector vote = new DoubleVector(this.ensemble_NL.getVotesForInstance(inst));
					if (vote.sumOfValues() > 0.0) {
						vote.normalize();
						vote.scaleValues(accuracy_NL / accuracySum);
						combinedVote.addValues(vote);
					}
				}
				to_return = combinedVote.getArrayRef();
				
				break;
			default:
				System.out.println("ERROR: getVotesForInstance()");
				break;
		}	
		return to_return;
	}
	
	private void clusteringModels() throws Exception {
		weka.core.Instances wekaInstances = this.instanceConverter.wekaInstances(this.predictionErrorByClassifierFromRepo);
			
		weka.filters.unsupervised.attribute.Remove filter = new weka.filters.unsupervised.attribute.Remove();
		filter.setAttributeIndices("" + (wekaInstances.classIndex() + 1));
				
		filter.setInputFormat(wekaInstances);
		weka.core.Instances wekaInstancesNoClass = weka.filters.Filter.useFilter(wekaInstances, filter);
				
		this.clusterer.buildClusterer(wekaInstancesNoClass);
				
		// Use forEachOrdered for debugging purposes.
		wekaInstancesNoClass.parallelStream().forEach(wekaInstNoClass -> {
			try {
				int clusterLabel = this.clusterer.clusterInstance(wekaInstNoClass);
				int instIndex = wekaInstancesNoClass.indexOf(wekaInstNoClass);
				this.predictionErrorByClassifierFromRepo.get(instIndex).setClassValue(clusterLabel);
				if (instIndex < this.repository.size()) {
					this.repository.get(instIndex).setClusterLabel(clusterLabel);
				} else {
					this.ensemble_NL.ensemble.get(0).setClusterLabel(clusterLabel);
				}
						
			} catch (Exception e) {
				e.printStackTrace();
			}
		});
	}
	
	private int getMostSimilarAndNewFromRepo(ClassifierWithInfo target) {
		
		if (this.repository.size() == 0) {
			return -1;
		}
		
		double[] qStatResults = new double[this.repository.size()];
		
		for (int i = 0; i < qStatResults.length; ++i) {
			qStatResults[i] = QStatistics.getQScoreForTwo(this.instWindow, target.getActualClassifier(), this.repository.get(i).getActualClassifier());
		}
		
		int maxQIndex = 0;
		for (int i = 1; i < qStatResults.length; ++i) {
			
			if (qStatResults[i] > qStatResults[maxQIndex]) {
				maxQIndex = i;
			} else if (qStatResults[i] == qStatResults[maxQIndex]) {
				maxQIndex = (this.repository.get(i).getActualClassifier().trainingWeightSeenByModel() 
								< this.repository.get(maxQIndex).getActualClassifier().trainingWeightSeenByModel()) ? i : maxQIndex;
			} else {
				/*
				 * Do nothing.
				 */
			}
		}
		
		return qStatResults[maxQIndex] <= this.similarityThreshold ? maxQIndex : -1;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		
		this.saveFIFO(this.instWindow, inst, this.windowSizeOption.getValue());
		
		double prediction = Utils.maxIndex(this.ensemble_NL.getVotesForInstance(inst)) == inst.classValue() ? 0.0 : 1.0;
		this.driftDetector.input(prediction);
		
		this.drift_level = DRIFT_LEVEL.NORMAL;
		if (this.driftDetector.getChange()) {
			this.drift_level = DRIFT_LEVEL.OUTCONTROL;
		}
		
		switch (this.drift_level) {
			case NORMAL:
				if (this.instSeenAfterDrift == this.windowSizeOption.getValue() && this.changeDetected > 0 && this.repository.size() > 0) {
	
					// determine the new model belongs to which cluster.
					// if CAN be determined: ensemble_NL = {C} ∪ {ClosestCluster.getModels(C, B)}
					// otherwise ensemble_NL = {C}
					
					this.initPredictionErrorStorage(this.instWindow.size());
					for (ClassifierWithInfo classifier : this.repository) {
						this.predictionErrorByClassifierFromRepo.add(classifier.makePredictionOnInstances(this.instWindow));
					}
					this.predictionErrorByClassifierFromRepo.add(this.ensemble_NL.ensemble.get(0).makePredictionOnInstances(this.instWindow));
			
					try {
						this.clusteringModels();
						
						int clusterToRecover = this.ensemble_NL.ensemble.get(0).getClusterLabel();
						
						List<ClassifierWithInfo> sortedRepo = new ArrayList<ClassifierWithInfo>(this.repository);
						sortedRepo.sort(Comparator.comparing(ClassifierWithInfo::getTrainingWeightSeenByModel));
						
						for (ClassifierWithInfo classifier : sortedRepo) {
							if (this.ensemble_NL.size() >= this.poolSizeOption.getValue()) {
								break;
							}
							if (classifier.getClusterLabel() == clusterToRecover) {
								this.ensemble_NL.add(classifier);
							}
						}

					} catch (Exception e) {
						e.printStackTrace();
					}
					
					this.predictionErrorByClassifierFromRepo.delete();
					this.resetClusterer();
					
					
				} else if (this.instSeenAfterDrift % this.windowSizeOption.getValue() == 0 && this.trainingHasStarted()) {

					if (this.ensemble_NL.size() >= this.poolSizeOption.getValue()) {
						
						// Get the worst model from ensemble_NL.
						ClassifierWithInfo worstInNL = this.ensemble_NL.removeWorst();
						
						if (this.repository.size() >= this.maxRepositorySize) {
							
							int mostSimilarIndex = this.getMostSimilarAndNewFromRepo(worstInNL);
							
		 					if (mostSimilarIndex > -1 &&
		 							worstInNL.getActualClassifier().trainingWeightSeenByModel() > this.repository.get(mostSimilarIndex).getActualClassifier().trainingWeightSeenByModel()) {
		 						
		 						this.repository.remove(mostSimilarIndex);
								worstInNL.resetPrequentialAccuracy();
								this.repository.add(worstInNL);
								
							} else {
								/**
								 * Do nothing, worstInNL will then be discarded.
								 */
							}

						} else {
							worstInNL.resetPrequentialAccuracy();
							this.repository.add(worstInNL);
						}
						
						
					}

					this.ensemble_NL.add(this.candidate);
					
					this.candidate = new ClassifierWithInfo(((Classifier) this.getPreparedClassOption(this.baseLearnerOption)).copy(),
							this.fadingFactorOption.getValue());
					
				} else {
					this.candidate.updatePrequentialAccuracy(inst);
					this.candidate.trainOnInstance(inst);
				}
				
				this.previous_drift_level = DRIFT_LEVEL.NORMAL;
				
				break;
				
			case OUTCONTROL:
				this.ensemble_OL = new EnsembleWithInfo(this.ensemble_NL);
				
				// Use NL because it will be clear afterwards, so can reset the prequential accuracy of the models without affecting OL 
				Boolean[] isAdd = new Boolean[this.ensemble_NL.size()];
				
				int tempMaxRepoSize = this.maxRepositorySize;
				
				for (int i = 0; i < this.ensemble_NL.size(); ++i) {
					
					if (this.repository.size() < tempMaxRepoSize) {
						isAdd[i] = true;
						--tempMaxRepoSize;
						continue;
					}
					
					int mostSimilarIndex = this.getMostSimilarAndNewFromRepo(this.ensemble_NL.getActualEnsemble().get(i));
					
					if (mostSimilarIndex > -1 &&
							this.ensemble_NL.getActualEnsemble().get(i).getActualClassifier().trainingWeightSeenByModel() > 
							this.repository.get(mostSimilarIndex).getActualClassifier().trainingWeightSeenByModel()) {
						
 						this.repository.remove(mostSimilarIndex);
 						isAdd[i] = true;
					} else {
						isAdd[i] = false;
					}
				}
				
				for (int i = 0; i < isAdd.length; ++i) {
					if (isAdd[i]) {
						ClassifierWithInfo toAdd = this.ensemble_NL.getActualEnsemble().get(i).copy();
						toAdd.resetPrequentialAccuracy();
						this.repository.add(toAdd);
					}
				}
				
				this.ensemble_NL.clear();
				
				this.ensemble_NH = new EnsembleWithInfo(this.fadingFactorOption.getValue(), false, "NH");
				
				if (this.previous_drift_level == DRIFT_LEVEL.NORMAL && this.repository.size() > 1) {
					this.candidate.resetLearning();
					
					// Do clustering
					// Create ensemble_NH
					this.initPredictionErrorStorage(this.instWindow.size());
					for (ClassifierWithInfo classifier : this.repository) {
						this.predictionErrorByClassifierFromRepo.add(classifier.makePredictionOnInstances(this.instWindow));
					}
						
					try {
						this.clusteringModels();
						
						int numOfClusters = this.clusterer.numberOfClusters();
						
						if (numOfClusters > 1) {
							
							// Get the most well-trained classifier from each cluster to form ensemble_NH.
							List<ClassifierWithInfo> sortedRepo = new ArrayList<ClassifierWithInfo>(this.repository);
							sortedRepo.sort(Comparator.comparing(ClassifierWithInfo::getTrainingWeightSeenByModel));
							
							
							for (int i = 0; i < sortedRepo.size() && numOfClusters > 0; ++i) {
								ClassifierWithInfo temp = sortedRepo.get(i);
								if (temp.getClusterLabel() == numOfClusters - 1) {
									this.ensemble_NH.add(temp);
									i = 0;
									numOfClusters -= 1;
								}
								if (i == sortedRepo.size() - 1) {
									i = 0;
									numOfClusters -= 1;
								}
							}
							
						} else {
							for (int i = 0; i < this.repository.size() && this.ensemble_NH.size() < this.poolSizeOption.getValue(); i += this.repositorySizeOption.getValue()) {
								this.ensemble_NH.add(this.repository.get(i));
							}
						}
							
					} catch (Exception e) {
						e.printStackTrace();
					}
							
					this.predictionErrorByClassifierFromRepo.delete();
					this.resetClusterer();
				}
				
				this.ensemble_NL = new EnsembleWithInfo(this.fadingFactorOption.getValue(), true, "NL");
				this.ensemble_NL.add(candidate);
				
				this.candidate = new ClassifierWithInfo(((Classifier) this.getPreparedClassOption(this.baseLearnerOption)).copy(),
														this.fadingFactorOption.getValue());
				
				this.ensemble_NH.resetPrequentialAccuracy();
				this.ensemble_NL.resetPrequentialAccuracy();
				this.ensemble_OL.resetPrequentialAccuracy();
				
				this.instWindow.clear();
				this.instSeenAfterDrift = 0;
				
				this.previous_drift_level = DRIFT_LEVEL.OUTCONTROL;
				
				changeDetected++;
				
				break;
			default:
				System.out.print("ERROR!");
				break;
		}
		
		if (this.ensemble_OL != null) {
			this.ensemble_OL.updatePrequentialAccuracy(inst);
		}
		if (this.ensemble_NH != null) {
			this.ensemble_NH.updatePrequentialAccuracy(inst);
		}
		this.ensemble_NL.updatePrequentialAccuracy(inst);
		this.ensemble_NL.trainOnInstance(inst);
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {

	}
	
	// Copied from WekaClusteringAlforithm.java
	private Class<?>[] findWekaClustererClasses() {
        AutoExpandVector<Class<?>> finalClasses = new AutoExpandVector<Class<?>>();
        Class<?>[] classesFound = AutoClassDiscovery.findClassesOfType("weka.clusterers",
                weka.clusterers.AbstractClusterer.class);
        for (Class<?> foundClass : classesFound) {
            finalClasses.add(foundClass);
        }
        return finalClasses.toArray(new Class<?>[finalClasses.size()]);
    }
	
	private void saveFIFO(List<Instance> buffer, Instance toAdd, int maxSize) {
		while (buffer.size() >= maxSize) {
			buffer.remove(0);
		}
		buffer.add(toAdd);
		this.instSeenAfterDrift++;
	}
	
	protected class EnsembleWithInfo extends AbstractClassifier {
		
		// TODO: For debugging
		private String name;
		
		private List<ClassifierWithInfo> ensemble;
		
		private double alpha;
		private double estimation;
		private double b;
		
		private boolean isWMEnsemble;
		
		protected EnsembleWithInfo(double prequentialAccFadingFactor, boolean isWMEnsemble, String name) {
			
			this.name = name;
			
			this.ensemble = new ArrayList<ClassifierWithInfo>();
			
			this.alpha = prequentialAccFadingFactor;
			this.estimation = 0.0;
			this.b = 0.0;
			
			this.isWMEnsemble = isWMEnsemble;
			
		}
		
		/*
		 * Copy Constructor
		 */
		protected EnsembleWithInfo(EnsembleWithInfo source) {
			
			if (source.name.equals("NL")) {
				this.name = "OL";
			}
			
			this.ensemble = new ArrayList<ClassifierWithInfo>(source.ensemble);
			
			this.alpha = source.alpha;
			this.estimation = source.estimation;
			this.b = source.b;
			
			this.isWMEnsemble = source.isWMEnsemble;
			
		}
		
		public EnsembleWithInfo copy() {
			return new EnsembleWithInfo(this);
		}
		
		protected int size() {
			return this.ensemble.size();
		}
		
		protected List<ClassifierWithInfo> getActualEnsemble() {
			return this.ensemble;
		}
		
		
		protected void clear() {
			this.ensemble.clear();
			this.estimation = 0.0;
			this.b = 0.0;
		}
		
		protected void add(ClassifierWithInfo toAdd) {
			this.ensemble.add(toAdd.copy());
		}
		
		protected ClassifierWithInfo removeWorst() {
			ClassifierWithInfo worst = this.ensemble
										   .stream()
										   .min(Comparator.comparingDouble(x -> x.getPrequentialAccuracy()))
										   .get();
			this.ensemble.remove(worst);
			
			return worst;
		}
		
		public double[] getVotesForInstance(Instance inst) {
			
			double accuracySum = this.ensemble
									 .parallelStream()
									 .mapToDouble(ClassifierWithInfo::getPrequentialAccuracy)
									 .sum();
			
			DoubleVector combinedVote = new DoubleVector();
			for (int i = 0; i < ensemble.size(); ++i) {
				if (ensemble.get(i).estimation > 0.0) {
					DoubleVector vote = new DoubleVector(ensemble.get(i).getVotesForInstance(inst));
						
					if (vote.sumOfValues() > 0.0) {
						vote.normalize();
						if (isWMEnsemble) {
							vote.scaleValues(ensemble.get(i).getPrequentialAccuracy() / accuracySum);
						}
						combinedVote.addValues(vote);
					}
				}
			}

			return combinedVote.getArrayRef();
			
		}
		
		protected void updatePrequentialAccuracy(Instance inst) {
			this.estimation = this.alpha * this.estimation +
							(Utils.maxIndex(this.getVotesForInstance(inst)) == (int) inst.classValue() ? 1.0 : 0.0);
			
			this.b = this.alpha * this.b + 1.0;

			this.ensemble
				.parallelStream()
				.forEach(commitee -> commitee.updatePrequentialAccuracy(inst));
		}
		
		protected double getPrequentialAccuracy() {
			return b > 0.0 ? this.estimation / this.b : 0.0;
		}
		
		protected void resetPrequentialAccuracy() {
			this.estimation = 0.0;
			this.b = 0.0;
			
			this.ensemble
				.parallelStream()
				.forEach(commitee -> commitee.resetPrequentialAccuracy());
		}

		@Override
		public boolean isRandomizable() {
			return false;
		}

		@Override
		public void resetLearningImpl() {
			// TODO Auto-generated method stub
			
		}

		@Override
		public void trainOnInstanceImpl(Instance inst) {
			this.ensemble
				.parallelStream()
				.forEach(commitee -> commitee.trainOnInstance(inst));
			
		}

		@Override
		protected Measurement[] getModelMeasurementsImpl() {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public void getModelDescription(StringBuilder out, int indent) {
			// TODO Auto-generated method stub
			
		}
		
	}
	
	protected class ClassifierWithInfo extends AbstractClassifier {
		
		private Classifier classifier;
		private int clusterLabel;
		
		private double alpha;
		private double estimation;
		private double b;
		
		protected ClassifierWithInfo(Classifier classifier, double prequentialAccFadingFactor) {
			this.classifier = classifier;		
			this.alpha = prequentialAccFadingFactor;
			
			this.resetLearning();
			
		}
		
		/*
		 * Copy Constructor
		 */
		protected ClassifierWithInfo(ClassifierWithInfo source) {
			this.classifier = source.classifier.copy();
			this.clusterLabel = source.clusterLabel;
			
			this.alpha = source.alpha;
			this.estimation = source.estimation;
			this.b = source.b;
		}
		
		public ClassifierWithInfo copy() {
			return new ClassifierWithInfo(this);
		}
		
		protected double getTrainingWeightSeenByModel() {
			return this.classifier.trainingWeightSeenByModel();
		}
		
		protected Classifier getActualClassifier() {
			return this.classifier;
		}
		
		protected void setClusterLabel(int label) {
			this.clusterLabel = label;
		}
		
		protected int getClusterLabel() {
			return this.clusterLabel;
		}
		
		public double[] getVotesForInstance(Instance inst) {
			return this.classifier.getVotesForInstance(inst);
		}		

		protected Instance makePredictionOnInstances(List<Instance> instances) {
			
			Instance predictions4Clustering = new DenseInstance(instances.size() + 1);
			
			predictions4Clustering.setDataset(predictionErrorByClassifierFromRepo);

			instances.parallelStream()
					 .forEach(inst -> predictions4Clustering.setValue(instances.indexOf(inst),
														this.classifier.correctlyClassifies(inst) ? 1.0 : 0.0));			
			predictions4Clustering.setMissing(predictions4Clustering.classIndex());
			
			return predictions4Clustering;
		}
		
		protected void updatePrequentialAccuracy(Instance inst) {
			this.estimation = this.alpha * this.estimation + (this.classifier.correctlyClassifies(inst) ? 1.0 : 0.0);
			this.b = this.alpha * this.b + 1.0;
		}
		
		protected double getPrequentialAccuracy() {
			return b > 0.0 ? this.estimation / this.b : 0.0;
		}
		
		protected void resetPrequentialAccuracy() {
			this.estimation = 0.0;
			this.b = 0.0;
		}

		@Override
		public boolean isRandomizable() {
			return false;
		}

		@Override
		public void resetLearningImpl() {
			this.classifier.resetLearning();
			this.clusterLabel = -1;
			
			this.resetPrequentialAccuracy();
		}

		@Override
		public void trainOnInstanceImpl(Instance inst) {
			this.classifier.trainOnInstance(inst);
		}

		@Override
		protected Measurement[] getModelMeasurementsImpl() {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public void getModelDescription(StringBuilder out, int indent) {
			// TODO Auto-generated method stub
			
		}
	
	}
	
	protected enum DRIFT_LEVEL {
		NORMAL, WARNING, OUTCONTROL
	}

}