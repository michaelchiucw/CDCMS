# Plots of the Time-Memory-Accuracy Analysis
This folder contains the all the plots of the Time-Memory-Accuracy Analysis in Section V of the paper.

CDCMS was evaluated prequentially with different hyper-parameter values that control its memory requirements and may affect its runtime, based on the five representative synthetic data streams that were analysed in Sections IV-C and IV-D in the paper: Sine1-(gradual, abrupt), Sine2-(gradual, abrupt), Agr3-gradual. We recorded the information about the average prequential accuracy, total runtime (CPU seconds) and model cost (RAM-hours) across thirty runs for each hyper-parameter combination per data stream. Each run was performed using the University of Birmingham's BlueBEAR HPC with 8 CPU cores and 8GB memory in a single computing node.
