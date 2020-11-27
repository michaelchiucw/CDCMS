# Concept Drift handling based on Clustering in the Model Space (CDCMS)
This repository contains the followings:
 - The MOA implementation of CDCMS
 - The analysis of time-accuracy-memory relationship in CDCMS
 - The statistical test results and the plots of the experiment results
 - Artifical datasets used in the experiment and hyper-parameter tuning of paper.

## Abstract
Data stream applications usually suffer from multiple types of concept drift. However, most existing approaches are only able to handle a subset of types of drift well, hindering predictive performance. We propose to use diversity as a framework to handle multiple types of drift. The motivation is that a diverse ensemble can not only contain models representing different concepts, which may be useful to handle recurring concepts, but also accelerate the adaptation to different types of concept drift. Our framework innovatively uses clustering in the model space to build a diverse ensemble and identify recurring concepts. The resulting diversity also accelerates adaptation to different types of drift where the new concept shares similarities with past concepts. Experiments with 20 synthetic and 3 real-world data streams containing different types of drift show that our diversity framework usually achieves similar or better prequential accuracy than existing approaches, especially when there are recurring concepts or when new concepts share similarities with past concepts.

#### Author
 - Chun Wai Chiu (Michael): cxc1015 at student dot bham dot ac dot uk
 - Leandro Minku: L dot L dot Minku at bham dot ac dot uk

#### Environment details
Java version: 11.0.1
MOA version: 2018.6.0
