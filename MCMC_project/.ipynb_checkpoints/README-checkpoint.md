# MCMC implementation for oil and gas production models

<div style="text-align: right"> Rui Kou, Wajahat Ali </div>
<div style="text-align: right"> 04/30/2019 </div>

#### 1. Introduction
Production Data Analysis has been particularly useful in forecasting the oil and gas production rate and estimating the ultimate recovery (EUR). The commonly used deterministic method relies heavily on the experience of the evaluator and provides little information on the uncertainty of the estimation. In this project, we apply the Bayesian probabilistic method to estimate the parameters in both empirical and physics-based models. We aim to achieve the following objectives: 
-  To develop a Python program that generates probabilistic decline curves using Bayesian statistics and the Markov Chain Monte Carlo sampling method
-  To investigate the performance of both empirical (Arps model) and physics-based decline curve models (e.g. Transient Hyperbolic Model, θ Function Model) using production history data from unconventional oil producers 

#### 2. Oil and Gas Production Data
In this project, we use real production data from 150 wells located in Midland Texas. The map below shows the location of wells. Note that each of the blue dot may represent around 10 wells close to each other. 
<p align="center">
<img src="./images/well_location.png" alt="drawing" width="300"/>
</p>

Oil production rate from a single well decines over the time. Depends of the formation property, operation condition and well completions quality, the decline curve could be smooth or noisy. We show the production history data of a representative well for 24 months. Our goal is to use the first 12 month data to build a proababilic model, and compare second year data with the P10-P90 interval. 

<img src="./images/24month_data.png" alt="drawing" width="350"/>


#### 3. Model Formulation
We first introduce the 3 decline curve models that we are going to implement. 

The first model is Arps model.Arps model is an impirical model, where the D parameter can be interpreted as the first order derivative line, and b parameter (constant) is the second derivative.

<img src="./images/Arps.png" alt="drawing" width="300"/>

To implement the probablistic Arps model, we need to estimate  3 parameters: $q_i, D_i,b$

The second model is Transient Hyperbolic Model (THM). This model was introduced by Fulfort and Blasingame in 2013. 
Different from traditional Arps model, they bring the physics meaning into the "b" parameter. 

<img src="./images/THM.png" alt="drawing" width="260"/>

Instead of using a constant b parameter, the THM model assums the flow was started in the "linear" flow regime, with a constant $b_i$ = 2.0. At the end of linear flow (at time $t_elf$), the b parameter smoothly transit to a constant value of b<1.0. One example of the changing b parameter is shown below. 

<img src="./images/b_THM.png" alt="drawing" width="300"/>
To implement the probablistic Arps model, we need to estimate  4 parameters: $q_i, D_i, b_f,t_{elf}$. 

The third model that we are implementing is the Jacobi θ Function model. This model was proposed by Gilding and Valko in 2018. This model is fully analytical based on solving a linear flow partial differential equation (PDE). The solution to the PDE is in the form of second Jacobi $\theta$ function.

<img src="./images/Jacobi_theta.png" alt="drawing" width="300"/>

After grouping the variable, we only need to estimate three parameters for the Jacobi $\theta$ function model, which are: $q_i, \chi, \eta$. 

#### 4. Probablistic Model Formulation
We use Arps model as an example to show our implementation of the probablistic model. The same formulation was applied to both THM and Jacobi $\theta$ models. 
We assume a non-informative prior distribution for each of the parameter. We pick normal distribution as the porposal distribution. The variance of the proposal normal distribution is selected in an trial and error manner. 

<img src="./images/prior.png" alt="drawing" width="400"/>

We assumed the likelyhood of observing a data point given the parameters follows normal distribution. This is illustrated in the figure below

<img src="./images/model_illustration.png" alt="drawing" width="300"/>
The likelihood function formulation is shown below: 

<img src="./images/likelihood.png" alt="drawing" width="250"/>

Finally, we show the acceptance ratio formulation.Since we are using a uniform distribution as prior distribution, the ratio of posterior distribution became the ratio of lieklyhood because constant prior. 

<img src="./images/accept.png" alt="drawing" width="400"/>


#### 5. Results
We implemented all three probablistic models in Python. Detailed code is provided in the current director. We take the last 1000 samples of the parameters and showed the fitted line below. 

<img src="./images/1000_samples.png" alt="drawing" width="700"/>

Also, we take the posterior mean of each parameter and plotted the fitted line together with original data. We also plotted the deterministic Arps model fitted by least square regression. 

<img src="./images/post_mean.png" alt="drawing" width="500"/>

We first plot the mixing and posterior distribution of parameters in the Arps model. 

<img src="./images/Arps_post.png" alt="drawing" width="500"/>

One can observe that the mixing of the chain for each parameter is good. Also, The posterior distribution of $D_i$, $q_i$ and $b$ is in the shape of normal distribution. Here the distribution of b is truncated for b values larger than $2.0$.  This is because the largest possible b value is 2, which is the case of transient linear flow. 

Next, we show the posterior distribution and mixing of THM model in the plot below. 

<img src="./images/THM_post.png" alt="drawing" width="500"/>

One can observe the mixing for $D_i$ and $q_i$ is good.And the distribution of $D_i$ and $q_i$ is also in normal distribution shape. However, the distribution and mixing of $b_f$ and $t_{elf}$ is very different. We argue this is because we train the model using only 12 months of production data. The "switching point", which is $t_{elf}$ usually happends during the second year. Thus, our current training data set has little or no information about $t_{elf}$ and $b_f$. This explaines why $b_f$ is evenly possible throughout the range of [0,1]. 

Lastly, we show the posterior distribution of parameters for the Jacobi $\theta$ model. 

<img src="./images/jacobi_post.png" alt="drawing" width="500"/>

Different from previous two models, the Jacobi $\theta$ models is very computationally expensive, because it requires a special function which has infinite summation term. We can observe the Markov chain of parameters are converging. However, the mixing parameters are not as good as previous models, becase we haven't found the best "step size" (variance) for the proposal distribution. Also, we only run the Markov chain until 10,000 samples, whereas previous models run upto 100,000 samples. 

To test the prediction performance of each of the model, we randomlly selected 9 wells and plotted the true second year production data and the P10, P90 prediction interval. 

<img src="./images/p_arps.png" alt="drawing" width="500"/>

<img src="./images/p_THM.png" alt="drawing" width="500"/>

<img src="./images/p_jacobi.png" alt="drawing" width="500"/>

Comparing the prediction of the three models, we have the following observation:
- For majority of the wells (6 out of 9), the uncertainty of 2nd year production is "roughly" captured by the p10-p90 interval. 
- Outlier/anomaly points exists in 2 out of 9 wells. The the prediction is largely influenced by the outlier.
- For both Arps and THM Jacobi $\theta$ models, the uncertainty (p10-p90 interval) increases for points further in the future. 
- For THM model, the uncertainty interval does not increase. This is because the initial b-parameter value are fixed at 2, until it reached the time of $t_{elf}$.

#### 6. Conclusion and Future Work

We implemented the probablistic decline curve using three models. As shown in the results section, the MCMC sampling method generate a reasonable uncertainty interval. However, there are many aspects that we need to improve on the current model:
- A outlier detection and replacement method is much in need. As shown in the 9 well comparison plots, one single outlier will drag the prediction below the true data. 
- Jacobi $\theta$ function model is very time consuming. It would be better if we can parallel the code and speed up the sampling process.
- It would also be useful to build a hierachical model. Because the parameters of neighbouring wells in the same oild field may be consider as sampled from a hyper-parameter distribution. Stan package should be incorporated into the hierachical model. 









