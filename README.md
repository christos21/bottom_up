# BUMP: Smart Grid Bottom-up Modeling and Power-flow Analysis

BUMP is a multi-purpose benchmarking testbed for low-voltage active distribution 
networks (ADNs). The testbed comprises a granular residential appliance-level dataset, 
a benchmarking framework based on quasi-static simulations, a set of technical 
indices and a non-intrusive load monitoring (NILM) tool. 

BUMP can model the load demand and generation profile of multiple LV prosumers 
forming an ADN based on a bottom-up modeling approach. Additionally, by providing 
the grid topology and line properties, quasi-static analysis is performed to assess 
the grid operation and identify benchmarking test cases. The necessary input/output 
communication is achieved via the filesystem. Furthermore, a pre-trained deep neural 
network (DNN) is adapted to near real-time NILM for DR services. 
An overview of BUMP and the core components is illustrated in the next figure.

<img src="general schema.png" width="600"/>

[comment]: <> (![]&#40;general schema.png&#41;)

Finally, a suite of benchmark case studies (including overvoltage, undervoltage and 
line congestion) is also included in this repository, supported by ancillary 
trouble-shooting services, such as NILM and load shifting.


### Notes
- Use the "bottom_up_packages.yml" file to create a virtual environment with all 
necessary dependencies.
- Install OpenDSS simulator.
- Run the script "dataset/create_csv_profiles.py" to extract the dataset from the mat file. 




