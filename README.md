# KMGCN
 In brain network analysis, individual-level data can provide biological features of individuals, while population-level data can provide demographic information of popula
tions. However, existing methods mostly utilize either individual- or population-level
 features separately, inevitably neglecting the multi-level characteristics of brain disorders. To address this issue, we propose an end-to-end multi-graph neural network model
 called KMGCN. This model simultaneously leverages individual- and population-level
 features for brain network analysis. At the individual level, we construct multi-graph
 using both knowledge-driven and data-driven approaches. Knowledge-driven refers
 to constructing a knowledge graph based on prior knowledge, while data-driven in
volves learning a data graph from the data itself. At the population level, we construct
 multi-graph using both imaging and phenotypic data. Additionally, we devise a pooling method tailored for brain networks, capable of selecting brain regions that impact
 brain disorders. We evaluate the performance of our model on two large datasets, ADNI
 and ABIDE, and experimental results demonstrate that it achieves state-of-the-art performance, with 86.87% classification accuracy for ADNI and 86.40% for ABIDE, ac
companied by around 10% improvements in all evaluation metrics compared to the
 state-of-the-art models. Additionally, the biomarkers identified by our model align
 well with recent neuroscience research, indicating the effectiveness of our model in
 brain network analysis and potential biomarker discovery. 

## Usage
Demonstration of the model training and testing process.
### Install dependencies
  pip install -r requirements.txt

### Download pre-processed ADNI dataset
<https://pan.baidu.com/s/1JWg4utjrSrTHptL7DGkBRw?pwd=1024 >

### Run the reasoning process
Training and testing models. Run `main.py`
