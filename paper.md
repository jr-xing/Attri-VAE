[![Elsevier](https://www.sciencedirect.com/us-east-1/prod/bd96d51d266808527bf1018bd38b59c0b4bc6286/image/elsevier-non-solus.svg)](https://www.sciencedirect.com/journal/computerized-medical-imaging-and-graphics "Go to Computerized Medical Imaging and Graphics on ScienceDirect")

[![Computerized Medical Imaging and Graphics](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122X00098-cov150h.gif)](https://www.sciencedirect.com/journal/computerized-medical-imaging-and-graphics/vol/104/suppl/C)

Under a Creative Commons [license](http://creativecommons.org/licenses/by/4.0/)

Open access

## Highlights

- Interpretable deep learning helps to understand clinical and imaging attributes.
- Latent space-based representations do not ensure control of data attributes.    
- Attribute-based methods help to explain different attributes in the latent space.
- Attention maps explain attribute encoding in regularized latent space dimensions.
    

## Abstract

[Deep learning](https://www.sciencedirect.com/topics/chemical-engineering/deep-learning "Learn more about Deep learning from ScienceDirect's AI-generated Topic Pages") (DL) methods where interpretability is intrinsically considered as part of the model are required to better understand the relationship of clinical and imaging-based attributes with DL outcomes, thus facilitating their use in the reasoning behind the medical decisions. Latent space representations built with [variational autoencoders](https://www.sciencedirect.com/topics/materials-science/variational-autoencoder "Learn more about variational autoencoders from ScienceDirect's AI-generated Topic Pages") (VAE) do not ensure individual control of data attributes. Attribute-based methods enforcing attribute disentanglement have been proposed in the literature for classical computer vision tasks in benchmark data. In this paper, we propose a VAE approach, the Attri-VAE, that includes an attribute regularization term to associate clinical and medical imaging attributes with different regularized dimensions in the generated latent space, enabling a better-disentangled interpretation of the attributes. Furthermore, the generated attention maps explained the attribute encoding in the regularized latent space dimensions. Using the Attri-VAE approach we analyzed healthy and [myocardial infarction](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/myocardial-infarction "Learn more about myocardial infarction from ScienceDirect's AI-generated Topic Pages") patients with clinical, cardiac morphology, and [radiomics](https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/radiomics "Learn more about radiomics from ScienceDirect's AI-generated Topic Pages") attributes. The proposed model provided an excellent trade-off between reconstruction fidelity, disentanglement, and interpretability, outperforming state-of-the-art VAE approaches according to several quantitative metrics. The resulting latent space allowed the generation of realistic synthetic data in the trajectory between two distinct input samples or along a specific attribute dimension to better interpret changes between different cardiac conditions.

-   [Previous](https://www.sciencedirect.com/science/article/pii/S0895611122001392)
-   [Next](https://www.sciencedirect.com/science/article/pii/S0895611122001410)

## Keywords

Deep learning

Interpretability

Attribute regularization

Variational autoencoder

Cardiac image analysis

## 1\. Introduction

[Deep learning](https://www.sciencedirect.com/topics/chemical-engineering/deep-learning "Learn more about Deep learning from ScienceDirect's AI-generated Topic Pages") (DL) methods have recently shown great success in many fields, from computer vision ([Pitale et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b52), [Zhu et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b74), [Goodfellow et al., 2014](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b23)) to [natural language processing](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/natural-language-processing "Learn more about natural language processing from ScienceDirect's AI-generated Topic Pages") ([Wu et al., 2019](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b70), [Deng and Liu, 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b19)), among numerous others. In addition, DL methods have started to dominate the medical imaging field ([Shen et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b66)), being used in a variety of medical imaging problems, such as segmentation of anatomical structures in the images ([Bernard et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b7), [Ronneberger et al., 2015](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b57), [López-Linares et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b43)), disease prediction ([Jo et al., 2019](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b30)), medical image reconstruction ([Higaki et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b26), [Kofler et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b34)) and clinical decision support ([Sanchez-Martinez et al., 2022](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b62)). Despite achieving exceptional results, DL methods [face](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/face "Learn more about face from ScienceDirect's AI-generated Topic Pages") challenges when applied to medical data regarding explainability, interpretability, and reliability because of their underlying black-box nature ([Singh et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b69), [McCrindle et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b47)). Hence, the need for tools that investigate interpretability in DL is also emerging in healthcare.

Recent reviews of interpretable DL can be found in [Singh et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b69), [Barredo Arrieta et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b5), [Molnar, 2022](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b48), [Masis, 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b46). Some methods have been proposed that employ backpropagation-based attention maps to either generate class activation maps that visualize the regions with high activations in specific units of the network ([Selvaraju et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b63)) or saliency maps using gradients of the inputs with respect to the outputs ([Simonyan et al., 2014](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b68), [Kapishnikov et al., 2019](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b31)). Other methods also proposed creating proxy models that focus on complexity reduction such as LIME ([Ribeiro et al., 2016](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b55)) or by approximating a value based on [game theory](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/game-theory "Learn more about game theory from ScienceDirect's AI-generated Topic Pages") optimal Shapley values to explain the individual predictions of a model ([Lundberg and Lee, 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b45)). However, it is key to design models that are inherently interpretable, rather than creating posthoc models to explain the black-box ones ([Rudin, 2019](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b59)).

Recently, models based on latent representations, such as [variational autoencoders](https://www.sciencedirect.com/topics/materials-science/variational-autoencoder "Learn more about variational autoencoders from ScienceDirect's AI-generated Topic Pages") (VAE), have become powerful tools in this direction ([Liu et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b41), [Biffi et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b8)), as their latent space is able to encode important hidden variables of the input data ([Kingma and Welling, 2014](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b33)). Especially, when dealing with data that contains different interpretable features (data attributes), it is interesting to see how and if these attributes have been encoded in the latent space. Even though the proposed approaches provide promising results, they have some limitations, one of which is that the encoded variables cannot be easily controlled; they mostly show an entangled behavior, meaning each latent factor maps to more than one aspect in the generative process ([Bengio et al., 2013](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b6)).

In order to bypass this limitation, much effort has been done to enforce disentanglement in the latent space ([Higgins et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b27), [Kim and Mnih, 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b32), [Rubenstein et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b58), [Chen et al., 2018a](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b16), [Chartsias et al., 2019](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b15)), being the majority of them unsupervised techniques ([Bengio et al., 2013](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b6), [Locatello et al., 2019](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b42)). While many of these methods show good disentanglement performance, they are not only sensitive to inductive biases (e.g., choice of the network, hyperparameters, or random seeds), but also some amount of supervision is necessary for learning effective disentanglement ([Locatello et al., 2019](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b42)). Moreover, since these methods are able to learn a factorized latent representation without attribute specification, they require a posthoc analysis to determine how different attributes are encoded to different dimensions of the latent space ([Pati and Lerch, 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b51)).

On the other hand, attribute-based methods aim to establish a correspondence between data attributes of interest and the latent space ([Hadjeres et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b25), [Lample et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b39), [Bouchacourt et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b9), [Pati and Lerch, 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b51)). However, these methods also have their drawbacks: some of them are limited to work only on certain types of data attributes ([Lample et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b39)); some impose additional constraints ([Bouchacourt et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b9)); very few of them are designed to work with continuous variables ([Hadjeres et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b25), [Pati and Lerch, 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b51)); some require differentiable computation of the attributes; and they are extremely sensitive to the hyperparameters ([Hadjeres et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b25)). However, [Pati and Lerch (2021)](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b51) have recently shown promising results for interpretability with their approach, associating each data attribute to a different regularized dimension of the latent space, which they have applied in the MNIST database for digit number recognition. The same approach was also employed as a post-processing step to generate interpretable and temporally consistent segmentations of [echocardiography](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/echocardiography "Learn more about echocardiography from ScienceDirect's AI-generated Topic Pages") images ([Painchaud et al., 2022](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b50)).

In this paper, we proposed a VAE-based approach (Attri-VAE), as it is able to encode certain hidden attributes of the data ([Carter and Nielsen, 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b12)) which can then be used to control data generation, and thus, improve the interpretation of the data attributes ([Pati and Lerch, 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b51)). Attri-VAE is an attribute-interpreter VAE based on attribute-based regularization ([Pati and Lerch, 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b51)) in the latent space, for an enhanced interpretation of clinical and imaging attributes obtained from multi-modal sources. Additionally, we also employed a classification network (MLP) that enables to identify different clinical conditions, e.g., healthy vs. pathological cases. Furthermore, we incorporate gradient-based attention map computation ([Liu et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b41)) to visually explain our proposed network by generating attention maps that show the high-response regions for each value of data attributes that are encoded in the regularized latent space dimensions. The main contributions of this work can be described as follows:

-   •
    
    The proposed approach is able to interpret different data attributes where specific ones are forced to be encoded along specific latent dimensions without the need for any posthoc analysis, while encouraging attribute disentanglement by employing β\-VAE as a [backbone](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/spine "Learn more about backbone from ScienceDirect's AI-generated Topic Pages") ([Higgins et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b27)).
    
-   •
    
    The structured latent space enables controllable data generation by changing the latent code of the regularized dimension (i.e., following the corresponding attribute), generating new data samples as a result of manipulating these dimensions. For instance, if the attribute represents the volume in a region of interest (ROI) and the corresponding regularized dimension is the first one of the latent code, then increasing values of this dimension would result in an increase in the ROI volume. The ability to generate a heterogeneous set of medical images is promising as collecting large annotated images is especially an issue in the medical image domain. As our approach allows the generation of diverse datasets, it may be applied in a variety of clinical scenarios. It could, for example, be used to train robust deep-learning models, or it may be trained with various clinical conditions to generate synthetic images by modifying the latent code of regularized dimensions that follows the corresponding data attribute. Furthermore, a controlled virtual cohort generation could also be employed to remove the burden of needing to gather and select real data for [clinical trials](https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/clinical-trial "Learn more about clinical trials from ScienceDirect's AI-generated Topic Pages").
    
-   •
    
    Attribute-based gradient-based attention maps provide a way to explain how the gradient information of individual attributes flow inside the proposed architecture by showing high-response regions in the images.
    
-   •
    
    The classification network provides a way to stratify different cohorts, based on the attributes in the latent space. In this way, the most discriminative features for the classification task are identified by projecting original samples into the latent space.
    

In this work, we have applied the proposed Attri-VAE approach to study cardiovascular pathological conditions, such as [myocardial infarction](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/myocardial-infarction "Learn more about myocardial infarction from ScienceDirect's AI-generated Topic Pages"), using the EMIDEC[<sup>1</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fn1) cardiac imaging dataset ([Lalande et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b36)), including clinical and imaging features, also exploring the association with [radiomics](https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/radiomics "Learn more about radiomics from ScienceDirect's AI-generated Topic Pages") descriptors. Additionally, we used ACDC MICCAI17 database[<sup>2</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fn2) as an external testing dataset.

![](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr1.jpg)

1.  [Download: Download high-res image (407KB)](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr1_lrg.jpg "Download high-res image (407KB)")
2.  [Download: Download full-size image](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr1.jpg "Download full-size image")

Fig. 1. Training framework of the proposed approach. Loss functions are shown in red arrows. The total loss function of the model is: L\=Lrecon+βLKL+LMLP+γLAR. (a) Losses computed for each data sample: multilayer perceptron (MLP) loss (LMLP), Kullback–Leibler (KL) loss (LKL), and reconstruction loss (Lrecon). (b) Attribute-regularization loss (LAR), computed inside a training batch that has n data samples. The input, a 3D image (X), first goes through the 3D convolutional encoder, qφ(Z|X), which learns to map X to the low dimensional space Z by outputting the mean (μ) and variance (σ) of the latent space distributions. The decoder, pθ(Xˆ|Z), then takes Z and outputs the reconstruction of the original input, (Xˆ). The predicted classes of the inputs, yc, are computed with a MLP module that consists of three fully connected (FC) layers. The corresponding MLP loss function is computed between yc and the ground truth label yGT. In (b), LAR is shown to regularize the first dimension of the latent space (Z1) with the attribute a1 (a1 and a2 represent the first and the second attributes, respectively). DistZ1 is the distance matrix of the first latent dimension, while Dista1 represents the distance matrix of the attribute a1.

The remainder of this paper is organized as follows. Firstly, we present the methodology and the details of our architecture in Section [2](https://www.sciencedirect.com/science/article/pii/S0895611122001288#sec2). We then describe the experimental setup and employed dataset in Section [3](https://www.sciencedirect.com/science/article/pii/S0895611122001288#sec3). Section [4](https://www.sciencedirect.com/science/article/pii/S0895611122001288#sec4) provides the results that are discussed in Section [5](https://www.sciencedirect.com/science/article/pii/S0895611122001288#sec5). Finally, in Section [6](https://www.sciencedirect.com/science/article/pii/S0895611122001288#sec6) we conclude our findings.

## 2\. Methodology

The overall structure of our framework is shown in [Fig. 1](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig1) (training) and [Fig. 2](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig2) (testing). The proposed Attri-VAE incorporates attribute regularization into a β\-VAE framework that was used as a [backbone](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/spine "Learn more about backbone from ScienceDirect's AI-generated Topic Pages") for the interpretation of data attributes. The trained network enables to generate new data samples by manipulating the data attributes, whereas the generated attribute-based attention maps explain how the gradient information of each attribute flows inside the proposed architecture. This section is organized firstly explaining the overall training criterion of the proposed model, with the following subsections describing each of the elements of our methodology and their integration.

![](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr2.jpg)

1.  [Download: Download high-res image (454KB)](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr2_lrg.jpg "Download high-res image (454KB)")
2.  [Download: Download full-size image](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr2.jpg "Download full-size image")

Fig. 2. The trained network can be used for: (a) latent space manipulation; and (b) generating attribute-based attention maps. For a given 3D data sample, X, the trained 3D convolutional encoder, qφ(Z|X), outputs the mean (μ) and variance (σ) vectors, then Z being sampled with the reparameterization trick. (a) Data generation process by changing only first (Z1) and second (Z2) regularized latent dimensions of Z, which correspond to two different data attributes (volume and maximum 2D diameter, respectively). Then, the decoder, pθ(X|Z), generates 3D outputs, X1 and X2, using the manipulated latent vectors, Z1 and Z2, respectively. (b) Attribute-based attention map generation for a given attribute, which is encoded in the first latent dimension (Z1). First, (Z1) is backpropagated to the encoder’s last convolutional layer to obtain the gradient maps (Grads1 and Grads2) with respect to the feature maps (F1 and F2). The gradient maps of (Z1) measure the linear effect of each pixel in the corresponding feature map on the latent values. After that, we compute the weights (w1 and w2) using global average pooling (GAP) on each gradient map. A [heat map](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/heat-map "Learn more about heat map from ScienceDirect's AI-generated Topic Pages") is generated by multiplying these values (w1,w2) with the corresponding feature map, summing them up and applying an activation unit (ReLU). Finally, the heat map is upsampled and overlaid with the input image to obtain the superimposed image (3D attention map). Additionally, the class score of the input, yc, is computed with the multilayer perceptron (MLP) that is connected to Z. Note that, in the figure it is assumed that the last convolutional layer of the encoder has 2 feature maps.

### 2.1. Training criterion

Attri-VAE is trained with a loss function, L, which is composed of four terms, as follows: (1)L\=Lrecon+βLKL+LMLP+γLAR.

The reconstruction loss, Lrecon, is based on the binary cross-entropy (BCE) between the input X and its reconstruction Xˆ, while the second term, LKL, employs the Kullback–Leibler (KL) divergence between the learned prior and the posterior distributions, weighted by a hyperparameter (β). An additional term, LMLP, estimates the BCE loss for the classification between the network prediction, yc, and the ground truth label, yGT. The final loss term, LAR, includes the attribute regularization, with a tunable hyperparameter (γ) that weights its [strength](https://www.sciencedirect.com/topics/materials-science/mechanical-strength "Learn more about strength from ScienceDirect's AI-generated Topic Pages"). In the following sections, detailed explanations of each loss term in our training criterion can be found (also see [Fig. 1](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig1)).

### 2.2. Variational autoencoder (VAE) and β\-VAE

A variational autoencoder ([Kingma and Welling, 2014](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b33)) is a generative model that consists of an encoder and a decoder, and aims to maximize the marginal likelihood of the reconstructed output, which is written as: (2)logpθ(X)≥EZ∼qφ(Z|X)\[logpθ(X|Z)\]−DKL(qφ(Z|X)∥p(Z))In this objective function, the first term is the log likelihood expectation that the input X can be generated by the sampled Z from the inferred distribution, qφ(Z|X). The second term corresponds to the KL divergence between the distribution of Z inferred from X, and the prior distribution of Z. Note that both distributions are assumed to follow a multivariate [normal distribution](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/normal-density-functions "Learn more about normal distribution from ScienceDirect's AI-generated Topic Pages").

In practice, the loss function of the [VAE](https://www.sciencedirect.com/topics/materials-science/variational-autoencoder "Learn more about VAE from ScienceDirect's AI-generated Topic Pages") consists of two terms: a first term that penalizes the reconstruction error between the input and output; and a second term forcing the learned distribution, qφ(Z|X), to be as similar as possible to the prior distribution, p(Z). In this case, the overall VAE loss can be written as: (3)LVAE(θ,φ)\=Lrecon(θ,φ)+LKL(θ,φ),where the reconstruction loss, Lrecon(θ,φ), and the KL loss, LKL(θ,φ), are computed as follows: (4)Lrecon(θ,φ)\=∑i\=1N‖Xˆ−X‖22, (5)LKL(θ,φ)\=DKL(qφ(Z|X)∥p(Z)).

In this work we chose to use β\-VAE as the backbone of our approach to encourage the disentanglement as it is easy to formulate and it has shown good performance based on one or more disentanglement metrics ([Higgins et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b27), [Burgess et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b10)).

The β\-VAE approach ([Higgins et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b27)) is an extension of the standard VAE that aims to learn a disentangled representation of the encoded variables in a completely unsupervised manner ([Locatello et al., 2019](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b42), [Higgins et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b27)) by simply giving more weight to the KL term, compared to the original VAE, with an extra hyperparameter β: (6)LVAE(θ,φ)\=Lrecon(θ,φ)+βLKL(θ,φ),

### 2.3. Attribute-based regularization

In order to better interpret the data attributes that are encoded in the latent space, we employ an attribute-based regularization loss ([Pati and Lerch, 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b51)), which aims at encoding encode an attribute a along a dimension d of the latent space (regularized dimension).

The attribute regularization loss, LAR, is calculated for the dimension d of the latent space in a training batch containing n training examples for the purpose of forcing the dimension d to have a monotonic relationship with the attribute values of a. The attribute regularization loss is then computed as follows: (7)LAR(d,a)\=MAE(tanh(δDistZd)−sgn(Dista)),where MAE is the mean absolute error, Dista is the attribute distance matrix, and DistZd is the distance matrix of the latent dimension d. These matrices are computed for all n data examples in the corresponding training batch, such that: (8)Dista\=a(Xi)−a(Xj), (9)DistZd\=Zid−Zjd,where i,j∈\[0,n), Xi and Xj are two exemplary samples (Eq. [(8)](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fd8)), and each D\-dimensional latent vector is represented as Z\={Zd}, where d∈\[0,D) (Eq. [(9)](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fd9)).

In Eq. [(7)](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fd7), tanh and sgn refer to hyperbolic tangent function and sign function, respectively, whereas δ is the hyperparameter that modulates the spread of the posterior distribution. For multiple selected attributes of interest to be encoded in the latent space, the overall loss function can be computed by summing all the corresponding objective functions together. Specifically, when the attribute set is A:{ak}, where k∈\[0,K) contains K attributes (K≤D, being D the latent size), then the overall loss function is computed as: (10)LAR\=∑k\=0K−1Ldk,ak,where dk represents the index of the regularized dimension for the attribute k. This process is represented in [Fig. 1](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig1)(b).

### 2.4. Classification network

Recently, performing a classification task using VAEs has been proposed to learn and separate different cohorts in the latent space. For example, [Biffi et al. (2020)](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b8) classified heart pathologies with [cardiac remodeling](https://www.sciencedirect.com/topics/pharmacology-toxicology-and-pharmaceutical-science/heart-ventricle-remodeling "Learn more about cardiac remodeling from ScienceDirect's AI-generated Topic Pages") using explainable task-specific shape descriptors learned directly with a VAE architecture from the input segmentations. Additionally, other approaches based on VAE have also been applied to analyze [coronary artery](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/epicardial-coronary-arteries "Learn more about coronary artery from ScienceDirect's AI-generated Topic Pages") diseases ([Clough et al., 2019](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b18)), Alzheimer’s disease ([Shakeri et al., 2016](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b64)) or to predict the response of [cardiomyopathy](https://www.sciencedirect.com/topics/pharmacology-toxicology-and-pharmaceutical-science/cardiomyopathy "Learn more about cardiomyopathy from ScienceDirect's AI-generated Topic Pages") patients to [cardiac resynchronization therapy](https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/cardiac-resynchronization-therapy "Learn more about cardiac resynchronization therapy from ScienceDirect's AI-generated Topic Pages") ([Puyol-Antón et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b53)).

In this line, to enforce class separation to the Attri-VAE, a [multilayer perceptron](https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/perceptron "Learn more about multilayer perceptron from ScienceDirect's AI-generated Topic Pages") (MLP) prediction network was connected to the latent vector, p(yc|Z) (see [Fig. 1](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig1)). The corresponding objective function can be computed as the binary cross entropy (BCE) between the network prediction yc and the ground truth label yGT, such that: (11)LMLP\=BCE(yc,yGT)

### 2.5. Attribute-based attention generation

The Attri-VAE facilitates data interpretation by generating new data samples as a result of scanning the regularized latent dimensions. Furthermore, it also provides a way to obtain attention maps from these dimensions (attribute-based attention map generation) for a better understanding on how gradient information of these attributes flows inside the proposed architecture (as can be seen in [Fig. 2](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig2)).

Attribute-based visual attention maps were generated by means of gradient-based computation (Grad-CAM) ([Selvaraju et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b63)), as proposed by [Liu et al. (2020)](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b41). Basically, a score is calculated from the latent space that is then used to estimate the gradients and attention maps. Specifically, given the posterior distribution inferred by the trained network for a data sample X, qφ(Z|X), the corresponding D\-dimensional latent vector Z is sampled using the reparameterization trick ([Kingma and Welling, 2014](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b33)). Subsequently, for a given attribute set A:{ak}, where k∈\[0,K) contains K attributes, attribute-based attention maps, Mdk, are generated for each regularized latent dimension Zdk by backpropagating the gradients to the encoder’s last convolutional feature maps (F:{Fi} where i∈\[0,n)): (12)Mdk\=ReLU(∑i\=1nwiFi),where dk is index of the regularized latent dimension for a given attribute k. The weights, wi, are computed using global average pooling (GAP), which allows us to obtain a scalar value, as follows: (13)wi\=GAP(∂Zdk∂Fi)\=1T∑p\=1j∑q\=1l(∂Zdk∂Fipq),where T\=j×l, (i.e., width×height), and Fipq is the pixel value at location (p,q) of the j×l matrix Fi. This process is visually summarized in [Fig. 2](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig2).

## 3\. Application for interpretable cardiology

### 3.1. Datasets

Initially, the EMIDEC dataset ([Lalande et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b36)) was used in our experiments. It is a publicly available database with delay-enhancement magnetic resonance images (DE-MRI) of 150 cases (100 and 50 cases for training and testing, respectively), with the corresponding clinical information. Each case includes a DE-MRI acquisition of the [left ventricle](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/left-ventricle "Learn more about left ventricle from ScienceDirect's AI-generated Topic Pages") (LV), covering from [base](https://www.sciencedirect.com/topics/pharmacology-toxicology-and-pharmaceutical-science/base "Learn more about base from ScienceDirect's AI-generated Topic Pages") to apex. The training set, with ground-truth segmentations, includes 67 [myocardial infarction](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/myocardial-infarction "Learn more about myocardial infarction from ScienceDirect's AI-generated Topic Pages") (MINF) cases and 33 [healthy subjects](https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/normal-human "Learn more about healthy subjects from ScienceDirect's AI-generated Topic Pages"). The testing set includes 33 MINF and 17 healthy subjects. Some clinical parameters were also provided along with the MRI: sex, age, tobacco (yes, no, and former), overweight, arterial hypertension, diabetes, family history of [coronary artery](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/epicardial-coronary-arteries "Learn more about coronary artery from ScienceDirect's AI-generated Topic Pages") disease, electrocardiogram (ECG), Killip max,[<sup>3</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fn3) [troponin](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/troponin "Learn more about troponin from ScienceDirect's AI-generated Topic Pages"),[<sup>4</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fn4) LV [ejection fraction](https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/heart-ejection-fraction "Learn more about ejection fraction from ScienceDirect's AI-generated Topic Pages") (EF), and NTproBNP.[<sup>5</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fn5) Furthermore, we also used an additional external testing dataset for a more robust assessment of the classification performance, the ACDC MICCAI17 challenge training dataset[<sup>6</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fn6) (end-diastole, ED, and end-systole, ES, cine-MRI from 20 healthy volunteers and 20 MINF cases). The ACDC dataset includes ground-truth segmentations of the left ventricle, [myocardium](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/myocardium "Learn more about myocardium from ScienceDirect's AI-generated Topic Pages"), and [right ventricle](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/right-ventricle "Learn more about right ventricle from ScienceDirect's AI-generated Topic Pages") by an experienced manual observer at both ED and ES time points ([Bernard et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b7)). The reader is referred to [Lalande et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b36), [Bernard et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b7) for more details on the MRI acquisition protocol.

As a pre-processing step, the intensities of the left ventricle in all images were scaled between 0 and 1. Additionally, each image was cropped and padded (x\=80; y\=80; z\=80; t\=1).

### 3.2. Cardiac attributes

Three different types of attributes were studied in our experiments. Initially, the Attri-VAE was trained with cardiac shape descriptors (e.g., wall thickness, LV and myocardial volumes, ejection fraction), extracted from ground-truth segmentations, which can easily be visually interpreted. In addition, attributes available in the clinical information with the highest discriminative performance were identified using recursive feature elimination (RFE) with a [support vector machine](https://www.sciencedirect.com/topics/chemical-engineering/support-vector-machine "Learn more about support vector machine from ScienceDirect's AI-generated Topic Pages") (SVM) classification model (linear kernel, regularization parameter C\=10) since this approach has already shown good performance for feature selection tasks ([Huang et al., 2014](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b28), [Samb et al., 2012](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b61), [Yang, 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b72)). In total 12 clinical attributes were provided with the EMIDEC dataset as introduced in Section [3.1](https://www.sciencedirect.com/science/article/pii/S0895611122001288#sec3.1) and the most discriminative attributes were then included in our analysis (e.g., gender, age, tobacco). The feature selection pipeline was done using the python-based machine learning library scikit-learn (version 1.0.2).[<sup>7</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fn7)

Finally, the Attri-VAE was also trained with [radiomics](https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/radiomics "Learn more about radiomics from ScienceDirect's AI-generated Topic Pages") features. Radiomics analysis was originally proposed to capture alterations at both the morphological and tissue levels in oncology applications ([Aerts et al., 2014](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b2), [Lambin et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b38)), deriving multiple quantifiable features from pixel-level data. More recently, radiomics approaches have provided promising results on cardiac MRI data, for discriminating different cardiac conditions ([Neisius et al., 2019](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b49), [Larroza et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b40), [Baessler et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b4), [Cetin et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b14)), and to study cardiovascular [risk factors](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/risk-factor "Learn more about risk factors from ScienceDirect's AI-generated Topic Pages") in large databases ([Cetin et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b13)). Radiomics analysis represents a step towards interpretability compared to other black-box approaches since some features can be related to pathophysiological mechanisms ([Cetin et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b13)). However, there is a need for improving the robustness and reproducibility of radiomics outcomes across different feature selection strategies and imaging protocols, which would lead to enhanced explainability. For this reason, radiomics features were employed in our experiments to benefit from the proposed network’s ability to explain the encoded attributes. The open-source library PyRadiomics (version 3.0.1)[<sup>8</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fn8) was used to derive 214 features per analyzed cardiac structure including 28 shape-based, 36 intensity-based, and 150 texture-based features. Subsequently, radiomics features with the highest discriminative performance were identified using the above-mentioned feature selection approach as this strategy has also demonstrated good performance in previous radiomics studies ([Zhang et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b73), [Xiao et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b71), [Chen et al., 2018b](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b17)). The top-performing features of this process were then selected to train the Attri-VAE.

### 3.3. Architectural details

The 3D convolutional encoder of the proposed Attri-VAE framework compresses the input into a 250-dimensional embedding through a series of 5 3D convolutional layers with kernel size 3 and stride 2, except the last convolutional layer that has stride 1. Note that, using a series of 3 fully connected layers, the latent dimension was set to 64. The prediction network was constructed with a shallow 3-layer MLP to be able to discriminate between the healthy and infarct subjects, using a ReLU activation function as a non-linearity after the first two layers. The upsampling and convolutional layers used in the encoder and the decoder were followed by batch normalization and ReLU non-linearity, except the decoder’s last convolutional layer (Attri-VAE output) where a sigmoid function was applied. All the network weights were randomly initialized with Xavier initialization ([Glorot and Bengio, 2010](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b22)). The tunable parameters of the loss function (Eq. [(1)](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fd1)) were fixed as follows: KL weight β\=2; and regularization weight γ\=200. Additionally, δ (Eq. [(7)](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fd7)) was set to 10. We provided detailed information on the model architecture of the proposed Attri-VAE, including our publicly available code, in our GitHub repository.[<sup>9</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fn9)

The Attri-VAE was trained on an NVIDIA Tesla T4 GPU using Adam optimizer with a learning rate equals to 0.0001 and batch size of 16 for 10,000 epochs. The dataset was split into 70/30 training (47 pathological, 23 healthy) and testing (20 pathological and 10 healthy subjects) sets. Subsequently, random oversampling of the normal subjects was employed in the training set as a strategy to treat the unbalanced behavior of the dataset; however, the testing set was kept unchanged. Note that the proposed model is implemented using python programming language and PyTorch library (version 1.10.0).[<sup>10</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fn10) Image pre-processing and transformations were done using the python-based MONAI library (version 0.8.0).[<sup>11</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fn11)

### 3.4. Experimental setting and evaluation criteria

The performance of the proposed Attri-VAE, both qualitatively and quantitatively, was compared with VAE, β\-VAE, and AR-VAE. As all three methods are the special cases of the proposed model, we demonstrated this comparison in several ablation studies by removing different components of the proposed network (i.e., β, MLP, and attribute-regularization (AR) components) such that VAE represents the removal of β, MLP and AR components of the Attri-VAE, β\-VAE represents Attri-VAE without MLP and AR losses and AR-VAE is Attri-VAE without MLP loss. First of all, the degree of disentanglement of the proposed latent space was evaluated with respect to different data attributes, using the following metrics available in the literature: the modularity metric, to analyze the dependence of each dimension of the latent space on only one attribute ([Ridgeway and Mozer, 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b56)) such that if the latent dimension is ideally modular then it will have high mutual information with a single attribute and zero mutual information with others ([Ridgeway and Mozer, 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b56)); the mutual information gap (MIG), to evaluate the MI difference between a given attribute and the top two dimensions of the latent space that share maximum MI with the corresponding attribute ([Chen et al., 2018a](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b16)); the separated attribute predictability (SAP), to measure the difference in the prediction error of the two most predictive dimensions of the latent space for a given attribute ([Kumar et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b35)); and the Spearman [correlation coefficient](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/correlation-coefficient "Learn more about correlation coefficient from ScienceDirect's AI-generated Topic Pages") (SCC) score, to compute its maximum value between an attribute and each dimension of the latent space.

In parallel, the interpretability metric introduced in [Adel et al. (2018)](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b1) was used to measure the ability to predict a given attribute using only one dimension of the latent space. For this, a linear probabilistic relationship between the corresponding data attribute and the regularized latent space dimension was calculated. Then, the interpretability score was computed by summing up the logarithms of these probabilities corresponding to each test sample ([Adel et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b1)). As for the β\-VAE model, dimensions having a high MI with the corresponding data attribute were chosen for the interpretability estimation. The reconstruction fidelity performance was also evaluated, employing the maximum mean discrepancy (MMD) score ([Gretton et al., 2007](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b24)), which measures the distance between the distributions of real and reconstructed data examples, as well as their mutual information (MI) as an image similarity metric. The interpretability and MI metrics were then used to identify the optimal values of the most relevant hyperparameters in Eq. [(1)](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fd1) and Eq. [(7)](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fd7) (i.e., β, γ, and δ), evaluating the influence of the KL divergence (β) and attribute regularization (γ) loss terms, as well as the weight of the distance matrix between two samples in a latent dimension. As a proof-of-concept, the hyperparameter sensitivity analysis was performed with only the four cardiac shape-based interpretable attributes.

Another set of experiments was carried out to explore the potential of the latent space generated by the Attri-VAE to create synthetically realistic samples. First, two samples in the Attri-VAE latent space, corresponding to input data with distinct cardiac characteristics (e.g., thin vs. thick myocardium, [absence](https://www.sciencedirect.com/topics/pharmacology-toxicology-and-pharmaceutical-science/epileptic-absence "Learn more about absence from ScienceDirect's AI-generated Topic Pages") vs. presence of myocardium infarct), were chosen as references to synthetically generate interpolated images through their trajectory. Secondly, we qualitatively evaluated the control over individual data attributes during the generation process of the Attri-VAE model. Given a sample with a latent code z, a given attribute (e.g., LV volume) can be scanned from low to high values changing the latent code of the corresponding regularized dimension, due to their monotonic relationship. The attribute scanning creates synthetically generated samples in a latent space trajectory where only the chosen attribute is changed, facilitating its interpretation. To further facilitate the identification of each attribute’s visual influence in the synthetically generated images, gradient-based attention maps were also estimated.

Finally, the performance of the Attri-VAE model for classifying healthy and pathological hearts was assessed using the area under the curve (AUC) and accuracy (ACC) metrics, using both the EMIDEC and the ACDC17 challenge datasets. The Attri-VAE results were benchmarked against other VAE-type approaches (VAE+MLP, β\-VAE+MLP), as well as to classical radiomics analysis (with SVM). The latent space projections of the Attri-VAE model, regularized by different attributes, were also qualitatively analyzed to identify the attributes better differentiating healthy and pathological clusters of samples.

## 4\. Results

### 4.1. Disentanglement and interpretability

The proposed Attri-VAE approach obtained better disentanglement metric scores than its ablated variants (i.e., β\-VAE and AR-VAE) using shape and clinical attributes, implying a more disentangled latent space.

Firstly, all models provided high modularity values (Attri-VAE: 0.98 vs. β\-VAE: 0.97 vs. 0.98 AR-VAE), signaling that each dimension of the latent spaces in all models only depended on one data attribute. The Attri-VAE also resulted in higher MIG/SAP scores than the others (Attri-VAE: 0.60/0.63 vs. β\-VAE: 0.02/0.05 vs. AR-VAE: 0.51/0.55). In its turn, the [SCC](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/correlation-coefficient "Learn more about SCC from ScienceDirect's AI-generated Topic Pages") metric estimated for Attri-VAE was substantially higher than the corresponding β\-VAE one and only slightly higher than the one for AR-VAE (Attri-VAE: 0.97 vs. β\-VAE: 0.46 vs. AR-VAE: 0.96) due to the monotonic relationship between a given attribute and the regularized latent dimension enforced by both Attri-VAE and AR-VAE. When using radiomics features, the same trend was observed, with some Attri-VAE disentanglement metrics (MIG and SAP) slightly lower than when using shape and clinical attributes (Attri-VAE / β\-VAE /AR-VAE): modularity, 0.98/0.98 /0.98; MIG, 0.49/0.01/0.49; SAP, 0.51/0.06/0.51; SCC, 0.98/0.42/0.97.

[Table 1](https://www.sciencedirect.com/science/article/pii/S0895611122001288#tbl1) shows the interpretability scores for Attri-VAE, β\-VAE, and AR-VAE obtained with clinical and shape descriptors as well as radiomics features. The radiomics feature selection identified seven of them having the most discriminative power: four shape-based, being the sphericity of the [left ventricle](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/left-ventricle "Learn more about left ventricle from ScienceDirect's AI-generated Topic Pages"), the maximum 2D diameter of the [myocardium](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/myocardium "Learn more about myocardium from ScienceDirect's AI-generated Topic Pages"), as well as left ventricle and myocardial volumes; three texture-based, being the correlation of the left ventricle, the difference entropy of the myocardium and the inverse variance of the left ventricle.

![](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr3.jpg)

1.  [Download: Download high-res image (450KB)](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr3_lrg.jpg "Download high-res image (450KB)")
2.  [Download: Download full-size image](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr3.jpg "Download full-size image")

Fig. 3. Three examples of real and reconstructed images using the VAE, β\-VAE, AR-VAE, and Attri-VAE approaches. Three slices are shown in every example: apical (APEX), mid-ventricle (MID), and basal (BASE) slices. Sample 1 and 3 correspond to healthy hearts while Sample 2 shows an infarcted myocardium.

Table 1. Interpretability score ([Adel et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b1)) of most relevant shape, clinical and radiomics attributes, as encoded in the latent space, with the Attri-VAE, β\-VAE, and AR-VAE approaches. LV: left ventricle, MYO: myocardium, [EF](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/ejection-fraction "Learn more about EF from ScienceDirect's AI-generated Topic Pages"): ejection fraction. Max: maximum, DE: difference entropy, ZP: zone percentage. Maximum interpretability is 1.0.

|   Empty Cell   | Attri-VAE | β\-VAE | AR-VAE |
|----------------|-----------|--------|--------|
|   LV volume    |   0.89    |  0.14  |  0.80  |
|   MYO volume   |   0.93    |  0.02  |  0.87  |
| Wall thickness |   0.95    |  0.10  |  0.90  |
|       EF       |   0.94    |  0.03  |  0.90  |
|     Gender     |   0.98    |  0.19  |  0.94  |
|      Age       |   0.93    |  0.12  |  0.84  |
|    Tobacco     |   0.70    |  0.19  |  0.74  |
|   Radiomics    |   0.91    |  0.06  |  0.90  |

It can easily be observed that the Attri-VAE provided a high degree of interpretability (i.e., close to 1.0) for all attributes, with the exception of tobacco (0.70). Among shape and [clinical features](https://www.sciencedirect.com/topics/pharmacology-toxicology-and-pharmaceutical-science/clinical-feature "Learn more about clinical features from ScienceDirect's AI-generated Topic Pages"), gender was the attribute with a higher interpretability (0.98), followed by the wall thickness (0.95), meaning that they could be predicted with only one dimension of the latent space. As for radiomics features, the average interpretability metric value was 0.91, with shape-based ones showing slightly larger values than texture features (0.93 and 0.89, respectively); the maximum 2D diameter of the myocardium presented the highest value (0.97). On the other hand, the β\-VAE clearly resulted in lower interpretability values (average of 0.11 for shape/clinical attributes and 0.06 for radiomics features) and AR-VAE obtained slightly lower interpretability metric score (average of 0.86 for shape/clinical attributes and 0.90 for radiomics features) than the proposed Attri-VAE.

### 4.2. Reconstruction fidelity

Reconstruction fidelity is another important factor as the high degree of disentanglement and interpretability should not be accompanied by a reduced reconstruction. Hence in this experiment, we demonstrated the reconstruction quality of the proposed approach by comparing its performance with its ablated variants, specifically VAE, β\-VAE, and AR-VAE.

The proposed Attri-VAE approach obtained the lowest MMD values, representing a lower distance between input and reconstructed images (see [Table 2](https://www.sciencedirect.com/science/article/pii/S0895611122001288#tbl2)). We observed that removing only the MLP loss from Attri-VAE (AR-VAE) obtained the best MI value of 0.92 (0.91 and 0.89 for VAE and Attri-VAE, respectively).

The reconstructions of three data examples from the EMIDEC dataset where the performance of the Attri-VAE was compared with its different variants, can be seen in [Fig. 3](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig3). Even though all models achieved similar qualitative reconstruction results, the Attri-VAE model generated images better preserving the heart shape and details than the other models: see the papillary muscles in mid-myocardium slices (dark regions in the blood pool) or the left ventricular cavity in apical slices of Sample 2 and Sample 3 in [Fig. 3](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig3). We can also observe in the figure that apical slices were more difficult to reconstruct than mid-ventricle and basal ones for the three tested models.

Table 2. Ablation study on the reconstruction accuracy of Attri-VAE on the EMIDEC dataset, quantified with the maximum mean discrepancy (MMD) and mutual information (MI) metrics. The MMD results are given as ± standard deviation. w/o : without, AR: attribute-regularization.

|       Empty Cell        |  MMD ×102   |  MI  |
|-------------------------|-------------|------|
| VAE (w/o β, MLP and AR) | 1.86 ± 0.06 | 0.91 |
| β\-VAE (w/o MLP and AR) | 1.38 ± 0.04 | 0.87 |
|    AR-VAE (w/o MLP)     | 1.74 ± 0.06 | 0.92 |
|        Attri-VAE        | 1.18 ± 0.03 | 0.89 |

### 4.3. Hyperparameter sensitivity analysis

This study evaluates the impact of hyperparameters on the interpretability and reconstruction quality of the Attri-VAE. The trade-off between reconstruction quality and interpretability can be seen in [Fig. 4](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig4) where the performance of β\-VAE (β = 3) is also provided for comparison. A visual inspection of the figure suggests that γ, i.e., the hyperparameter controlling the attribute regularization, was the key to obtaining good interpretability values while keeping reasonable reconstruction fidelity (mutual information ≥ 0.88), with values of γ≥ 100.

![](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr4.jpg)

1.  [Download: Download high-res image (133KB)](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr4_lrg.jpg "Download high-res image (133KB)")
2.  [Download: Download full-size image](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr4.jpg "Download full-size image")

Fig. 4. Effect of hyperparameters on the interpretability and reconstruction fidelity of the Attri-VAE approach. The hyperparameters β and γ of the Attri-VAE model control the influence of the loss terms for the Kullback–Leibler divergence between learned prior and posterior distributions, and attribute regularization, respectively. Each marker represents a unique combination of the hyperparameters β and γ, which is indicated by color and shape, respectively. For comparison, the performance of β\-VAE (β = 3) is also represented. Best performance combinations are located in the top right corner of the graph.

On the other hand, the β hyperparameter was not as relevant as the γ hyperparameter. As expected, the β\-VAE provided acceptable reconstruction fidelity results but low values of interpretability. We need to point out that the same results were obtained when using radiomics features instead of shape-based attributes.

### 4.4. Latent space interpolation and attribute scanning

This experiment aims to qualitatively evaluate the proposed latent space by demonstrating its ability to interpolate between different data examples and control individual attributes during the generation process.

For the first experiment, the interpolation was employed, between distinct and well-separated samples in the learned latent space of the Attri-VAE. The proposed approach generates synthetic interpolated images that have a realistic appearance, gradually changing the main sample characteristics in the trajectory between the chosen samples. The first row of [Fig. 5](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig5) clearly demonstrates the Attri-VAE model’s ability to create smooth transitions between hearts having largely different characteristics such as (thin to thick) wall thickness. The other two rows of the figure demonstrate a similar behavior from non-infarcted/scar to infarcted/scar patients.

![](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr5.jpg)

1.  [Download: Download high-res image (167KB)](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr5_lrg.jpg "Download high-res image (167KB)")
2.  [Download: Download full-size image](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr5.jpg "Download full-size image")

Fig. 5. Linear latent space interpolation between two data samples (extremes of each row in yellow frames) from the EMIDEC dataset. Each row depicts the interpolation from the left to the right latent vector dimension. Top: from thin to thick myocardium. Middle: from a myocardium with a scar to one without. Bottom: from a [healthy subject](https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/normal-human "Learn more about healthy subject from ScienceDirect's AI-generated Topic Pages") to a patient with a myocardial infarct.

![](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr6.jpg)

1.  [Download: Download high-res image (1005KB)](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr6_lrg.jpg "Download high-res image (1005KB)")
2.  [Download: Download full-size image](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr6.jpg "Download full-size image")

Fig. 6. Scanning of attributes and corresponding gradient-based attention maps for shape and radiomics features. The image in the middle (4th column, in yellow frame) shows the original reconstructed image. DE: difference entropy, IV: inverse variance, Max 2D dia: maximum 2-dimensional diameter, LV: left-ventricle, MYO: myocardium. Note that the first three rows demonstrate the attribute scanning that was done on the latent space of Attri-VAE, which was trained with clinical and shape features. The remaining rows represent the attribute scanning on the latent space of Attri-VAE trained with selected radiomics features.

The second experiment presents the effect of scanning an individual attribute along its corresponding regularized dimension in the Attri-VAE model, where all the remaining attributes remain fixed. The first three rows of [Fig. 6](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig6) exemplify the attribute scanning that was done on the latent space of Attri-VAE, which was trained with clinical plus shape features. The rest of the rows represent the attribute scanning on the latent space of Attri-VAE trained with selected radiomics features. For shape-based attributes, the changes in the attribute when moving along different values of the regularized dimension are clearly seen. For instance, from the left to the right in [Fig. 6](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig6), how LV and myocardial volumes are increasing in the first and second rows, respectively, or how the LV becomes more spherical. More subtle changes are observed with texture-based radiomics but they can still be identified with a careful inspection of the generated images. For example, moving along the latent space dimension corresponding to the correlation LV, we find more or less intensity homogeneity in the LV. The LV inverse variance (LV-IV) and the difference entropy of the myocardium (DE-MYO) only produced small changes that consisted in slightly thicker myocardium with lower values of LV-IV (left samples in [Fig. 6](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig6)) and some darker patches and heterogeneous texture in the myocardium for higher values of DE-MYO (right in [Fig. 6](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig6)). It needs to be pointed out that attribute scanning for clinical attributes such as age, gender, and tobacco is not shown since the images do not visually change along the corresponding regularized dimensions.

To further illustrate the impact of each studied attribute, we have also generated the gradient-based attention maps linked to the changes in each regularized latent dimension of the Attri-VAE model. Attention maps show the high response regions in the images when changing specific dimensions of the latent space that correspond to specific attributes. We can see in [Fig. 6](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig6) that more attention (i.e., higher response) is paid to more varying regions for shape-based attributes (e.g., right side of the slide for LV volume, where LV is increasing from the left to the right in the regularized dimension). In general, attention maps for texture-based features have less high-response regions than for shape-based attributes. However, in some texture-based features such as the difference entropy of the myocardium, a higher response can still be localized (in this example, darker regions in the top left part of the slice). On the other hand, interpretation and validation of the resulting attention maps for other attributes such as for LV-IV are more challenging.

### 4.5. Classification

This experiment illustrates the classification performance of the proposed Attri-VAE. Additionally, we have also compared its performance with several models, such as radiomics analysis and its ablated versions. The best result was obtained in both EMIDEC and ACDC datasets with the Attri-VAE trained with radiomics features (accuracy of 0.97 and 0.58 for both datasets), while radiomics+MLP was the worst for EMIDEC (accuracy of 0.76) and the β\-VAE+MLP for ACDC (accuracy of 0.45) (see [Table 3](https://www.sciencedirect.com/science/article/pii/S0895611122001288#tbl3)). There were only minor differences in the accuracy of the Attri-VAE method when trained with clinical and shape attributes or radiomics features. All the evaluated models, trained with the EMIDEC data, substantially dropped their performance when tested on the external ACDC dataset, especially the VAE-based approaches.

Finally, the latent space projections of different regularized latent dimensions are visualized in [Fig. 7](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig7), with plot axes representing the encoded data attributes. As it can be observed in the figure, our model is able to build several reduced dimensionality spaces, based on different attributes, where healthy and pathological cases (red and blue in the figure, respectively) can easily be clustered. For instance, the maximum 2D diameter of the myocardium and the LV volume attributes correctly separate most samples into two clusters. Interestingly, despite Attri-VAE having poor control over clinical attributes such as age or gender, they also facilitate the construction of the latent spaces and sample discrimination, as can be seen in the gender-age plot of [Fig. 7](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig7).

Table 3. Ablation study on the classification performance of Attri-VAE with EMIDEC and ACDC datasets (healthy vs. myocardial infarction) with different models. The results are reported as accuracy/AUC scores. [SVM](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/support-vector-machine "Learn more about SVM from ScienceDirect's AI-generated Topic Pages"): support vector machine, w/o : without, AR: attribute-regularization.

|         Empty Cell         |  EMIDEC   |   ACDC    |
|----------------------------|-----------|-----------|
| Attri-VAE (Clinical+Shape) | 0.93/0.94 | 0.55/0.54 |
|   Attri-VAE (Radiomics)    | 0.97/0.96 | 0.58/0.52 |
|    β\-VAE+MLP (w/o AR)     | 0.90/0.90 | 0.45/0.31 |
|   VAE+MLP (w/o β and AR)   | 0.87/0.80 | 0.53/0.35 |
|  Radiomics analysis (SVM)  | 0.77/0.75 | 0.60/0.61 |
|  Radiomics analysis (MLP)  | 0.76/0.75 | 0.58/0.60 |

![](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr7.jpg)

1.  [Download: Download high-res image (332KB)](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr7_lrg.jpg "Download high-res image (332KB)")
2.  [Download: Download full-size image](https://ars.els-cdn.com/content/image/1-s2.0-S0895611122001288-gr7.jpg "Download full-size image")

Fig. 7. Latent space projections of regularized dimensions for different clinical, shape, and radiomics attributes. Each point in the graphs represents a healthy or a myocardial infarction patient (red and blue, respectively), LV: left-ventricle, MYO: myocardium, IV: inverse variance, DE: difference entropy, Max 2D dia: maximum 2-dimensional diameter.

## 5\. Discussion

The analysis of medical data demands for interpretable methods. However, the majority of [deep learning](https://www.sciencedirect.com/topics/chemical-engineering/deep-learning "Learn more about deep learning from ScienceDirect's AI-generated Topic Pages") methods do not fulfill the minimum level of interpretability to be used in reasoning medical decisions ([Sanchez-Martinez et al., 2022](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b62)), being difficult to relate clinically and physiologically meaningful attributes with model parameters and outcomes. Fortunately, interpretable and explainable deep learning methods are starting to emerge. Models creating latent space representations, such as variational autoencoders, are promising but attributes are usually entangled in the resulting reduced dimensionality space, hampering its interpretation. In this work, we have presented the Attri-VAE approach that generates disentangled and interpretable representations where different types of attributes (e.g., clinical, shape, radiomics) are individually encoded into a given dimension of the resulting latent space. The results obtained by the proposed Attri-VAE model based on disentanglement and interpretability metrics clearly outperformed the state-of-the-art β\-VAE approach and obtained slightly better results than AR-VAE, indicating a high degree of disentanglement and a monotonic relationship between a given attribute and the corresponding regularized dimension. However, Attri-VAE values for some metrics such as the MIG and SAP, although substantially better than those of β\-VAE, were far from the maximum (e.g., 1.0). The same trend was observed by [Pati and Lerch (2021)](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b51) in the MNIST (i.e., for digit number identification) dataset, suggesting that other latent dimensions, beyond the regularized ones, share a high MI with different attributes. We would like to point out here that the performance of Attri-VAE was compared with several methods including, AR-VAE ([Pati and Lerch, 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b51)), further work is needed to compare the performance of the proposed approach with additional models, such as Guided-VAE ([Ding et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b20)). More work is also required to visually evaluate the model interpretability by human experts in addition to the quantitative results, in order to better illustrate the interpretation performance of the proposed approach.

Hyperparameter selection was a key step for finding the optimal Attri-VAE configuration providing an excellent trade-off between reconstruction fidelity, at the level of state-of-the-art alternatives, and interpretability; even though the Attri-VAE approach had a more constrained latent space, it generated reconstructions that are less smooth than other VAE-based models and more similar to the original input images. This explains why the proposed Attri-VAE had a slightly lower MI value and higher MMD and interpretability values than the other models. The most critical parameter to enforce interpretability was the weight of the attribute regularization loss term (γ in Eq. [(1)](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fd1)). The Attri-VAE plot of reconstruction fidelity vs. interpretability, shown in [Fig. 4](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig4) had the same pattern as the one obtained by [Pati and Lerch (2021)](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b51). Interestingly, their optimal γ values were lower than ours (\[5.0, 10.0\] vs ≥ 100), likely due to the higher complexity of the cardiac MRI data and corresponding latent space compared to the MNIST dataset. On the other hand, the best δ values were the same in the two studies (\[1.0, 10.0\]). However, it is worth noting that we did not evaluate the effect of the size of the latent space and the position of regularized dimensions. Therefore, future work is needed to analyze the size of the latent space and the importance of the position of regularized dimensions by, for example, randomly replacing them during inference. Additionally, more work is required to determine how various losses (i.e., classification and attribute losses) affect the latent space as a whole and if this is dependent on the latent space’s dimensionality.

One of the most interesting characteristics of the Attri-VAE approach is the ability to create realistic synthetic data by sampling the created latent space and interpolating between different original reconstructed inputs, which can be very useful for controllable and attribute-based data augmentation of training datasets in machine learning applications. Scanning a regularized dimension of the latent space creates synthetic images where the corresponding attribute changes its values, as can easily be observed for shape descriptors (e.g., LV and myocardial volumes, wall thickness) in [Fig. 6](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig6). In addition, the proposed approach allows a better understanding of some (texture-based) radiomics features, which are often difficult to interpret. However, clinical attributes such as age, gender, or tobacco consumption, despite obtaining good interpretability scores, did not create visually different interpolated samples over the regularized dimensions. One potential reason is the difficulty of the attribute regularization to control binary attributes, as suggested by [Pati and Lerch (2021)](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b51). Furthermore, the studied clinical attributes cannot be disassociated from the shape and image intensity variations (e.g., morphological changes of the heart with age), thus it is too restrictive to keep all attributes fixed except a clinical one. In consequence, we have employed the clinical attribute referring to healthy vs. myocardial infarction as a task-specific label, as the aim of this classification is to be able to separate different groups of patients in the latent space, by enforcing the continuous variables to be able to predict the desired class and provide interpretable results. More work is needed to better construct latent spaces where clinical information can be disentangled from other attributes. Additionally, we would like to point out that the proposed Attri-VAE was trained on delineated left ventricle images; future work needs to include other cardiac structures to integrate global changes in cardiac tissue and shape.

The generated gradient-based attention maps contributed to locally identifying the cardiac regions where the attributes were influencing, which was particularly useful for global attributes and for complex features such as the texture ones. Additionally, we only employed the well-known Grad-CAM method, which could be complemented with additional interpretability methods (e.g., LIME and its variations ([Ribeiro et al., 2016](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b55))) to better understand the attribute effects on the latent space. However, the attention maps have some limitations that have already been addressed in different studies in the literature, which demonstrate that saliency maps underperform in some key criteria such as localization, parameter sensitivity, repeatability, and reproducibility ([Arun et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b3)). Thus the reliability of attention maps still requires further investigation to assess its robustness and reliability with respect to data input and model parameter perturbations ([Reyes et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b54)). In parallel, enhanced 3D visualizations of the generated samples are needed to have an overall perspective of the cardiac differences, beyond 2D slice views of the resulting images.

The proposed Attri-VAE model also achieved excellent classification performance (healthy vs. myocardial infarction), outperforming the other VAE-based approaches, with slightly better results when trained with radiomics. When evaluated with the EMIDEC training dataset using ground-truth labels, the Attri-VAE approach provided accuracy results (0.98) equivalent to the best challenge participants reporting their performance on the same dataset (1.0 ([Lourenço et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b44)), 0.95 ([Shi et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b67)), 0.94 ([Ivantsits et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b29)) and 0.90 ([Sharma et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b65))). For the testing EMIDEC dataset ([Lalande et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b37)), the best participant method obtained a decreased accuracy (0.82, ([Lourenço et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b44), [Girum et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b21))), increasing to 0.92 for the challenge organizers ([Shi et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b67)). As for the ACDC dataset, which was tested as an external database (i.e., without considering it in training), classification accuracy was substantially reduced (0.58), being worst than results reported by challenge participants ([Bernard et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b7)) (0.96) to classify between the different pathologies (not only between healthy and myocardial infarction). On the other hand, we would like to point out that these two datasets have employed different image acquisition techniques and contain images from different imaging modalities, such that the EMIDEC dataset consists of DE-MRI images and the ACDC dataset contains cine-MRI images. However, further work is required to find out the main reason behind this performance drop including evaluating the Attri-VAE’s performance in comparison with an additional model, such as a baseline CNN. Additionally, more work is also needed to improve the reconstruction quality and the generalization of the Attri-VAE model to unseen data, being more robust to different quality and imaging acquisition protocols, through domain adaptation techniques or image registration or integrating these differences into its latent space, using databases such as the M&Ms challenge ([Campello et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b11)).

One limitation of the Attri-VAE approach, also acknowledged by [Pati and Lerch (2021)](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b51), is the dependence on the selection of the data attributes to train the model. An incorrect attribute selection could lead to undesired strong correlations of several attributes that will not ensure a monotonic relationship with the corresponding regularized dimension, leading to less attribute interpretation and reconstruction quality. However, the projection of original samples in latent spaces with regularized dimensions for different attributes (see [Fig. 7](https://www.sciencedirect.com/science/article/pii/S0895611122001288#fig7)) could be used as an interpretable attribute selection, identifying the ones better separating the analyzed classes such as the maximum 2D diameter of the myocardium and the LV volume attributes in our experiments. Further work will focus on fully integrating advanced feature selection techniques with the Attri-VAE model, as well as exploring alternative interpretability methods (see the recent review of [Salahuddin et al. (2022)](https://www.sciencedirect.com/science/article/pii/S0895611122001288#b60)) to better understand the role of clinical and imaging attributes on medical decisions in [cardiovascular applications](https://www.sciencedirect.com/topics/materials-science/cardiovascular-application "Learn more about cardiovascular applications from ScienceDirect's AI-generated Topic Pages"). Self-supervision will also be explored, as an opportunity to make use of unlabeled data to further improve our results.

## 6\. Conclusions

We have presented an approach, referred to as Attri-VAE, which implements attribute-based regularization in a β\-VAE scheme with a classification module for the purpose of attribute-specific interpretation, synthetic data generation, and classification of cardiovascular images. The basis of the proposed Attri-VAE model is to structure its latent space for encoding individual data attributes to specific latent dimensions, being guided by an attribute regularization loss term. The resulting constrained latent space can be easily manipulated along its regularized dimensions for an enhanced interpretation of different attributes. Additionally, the proposed approach improves the current state-of-the-art for classifying cardiovascular images and allows the visualization of the most discriminative attributes by projecting the trained latent space. Future work will be focused on improving the generalization of the trained Attri-VAE model to images with different acquisition characteristics.

## Funding

This work was partly funded by the European Union’s Horizon 2020 research and innovation programme under grant agreement No [825903](https://www.sciencedirect.com/science/article/pii/S0895611122001288#GS1) (euCanSHare project).

## Ethical approval

## CRediT authorship contribution statement

Irem [Cetin](https://www.sciencedirect.com/topics/pharmacology-toxicology-and-pharmaceutical-science/cetirizine "Learn more about Cetin from ScienceDirect's AI-generated Topic Pages"): Conceptualization, Methodology, Formal analysis, Software, Investigation, Writing – original draft, Writing – review & editing. **Maialen Stephens:** Writing – original draft, Writing – review & editing. **Oscar Camara:** Conceptualization, Methodology, Formal analysis, Writing – original draft, Writing – review & editing, Supervision. **Miguel A. González Ballester:** Conceptualization, Methodology, Formal analysis, Writing – original draft, Writing – review & editing, Supervision.

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

## References

1.  [Adel et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb1)
    
    Discovering interpretable representations for both deep generative and discriminative models
    
    Dy J., Krause A. (Eds.), Proceedings of the 35th International Conference on Machine Learning, Proceedings of Machine Learning Research, vol. 80, PMLR (2018), pp. 50-59
    
2.  [Aerts et al., 2014](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb2)
    
    Aerts H.J., Velazquez E.R., Leijenaar R.T.H., Parmar C., Grossmann P., Cavalho S., Bussink J., Monshouwer R., Haibe-Kains B., Rietveld D.H.F., Hoebers F.J., Rietbergen M.M., Leemans C.R., Dekker A., Quackenbush J., Gillies R.J., Lambin P.
    
    Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach
    
3.  [Arun et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb3)
    
    Arun N., Gaw N., Singh P., Chang K., Aggarwal M., Chen B., Hoebel K., Gupta S., Patel J., Gidwani M., Adebayo J., Li M.D., Kalpathy-Cramer J.
    
    Assessing the trustworthiness of saliency maps for localizing abnormalities in medical imaging
    
4.  [Baessler et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb4)
    
    Baessler B., Luecke C., Lurz J., Klingel K., von Roeder M., de Waha S., Besler C., Maintz D., Gutberlet M., Thiele H., Lurz P.
    
    Cardiac MRI texture analysis of T1 and T2 maps in patients with infarctlike acute myocarditis
    
5.  [Barredo Arrieta et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb5)
    
    Barredo Arrieta A., Díaz-Rodríguez N., Del Ser J., Bennetot A., Tabik S., Barbado A., Garcia S., Gil-Lopez S., Molina D., Benjamins R., Chatila R., Herrera F.
    
    Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI
    
6.  [Bengio et al., 2013](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb6)
    
    Bengio Y., Courville A., Vincent P.
    
    Representation learning: A review and new perspectives
    
    IEEE Trans. Pattern Anal. Mach. Intell., 35 (8) (2013), pp. 1798--1828, [10.1109/tpami.2013.50](https://doi.org/10.1109/tpami.2013.50)
    
7.  [Bernard et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb7)
    
    Bernard O., Lalande A., Zotti C., Cervenansky F., Yang X., Heng P.-A., Cetin I., Lekadir K., Camara O., Gonzalez Ballester M.A., Sanroma G., Napel S., Petersen S., Tziritas G., Grinias E., Khened M., Kollerathu V.A., Krishnamurthi G., Rohé M.-M., Pennec X., Sermesant M., Isensee F., Jäger P., Maier-Hein K.H., Full P.M., Wolf I., Engelhardt S., Baumgartner C.F., Koch L.M., Wolterink J.M., Išgum I., Jang Y., Hong Y., Patravali J., Jain S., Humbert O., Jodoin P.-M.
    
    Deep learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: Is the problem solved?
    
8.  [Biffi et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb8)
    
    Biffi C., Cerrolaza J.J., Tarroni G., Bai W., de Marvao A., Oktay O., Ledig C., Folgoc L.L., Kamnitsas K., Doumou G., Duan J., Prasad S.K., Cook S.A., O’Regan D.P., Rueckert D.
    
    Explainable anatomical shape analysis through deep hierarchical generative models
    
9.  [Bouchacourt et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb9)
    
    Bouchacourt D., Tomioka R., Nowozin S.
    
    Multi-level variational autoencoder: Learning disentangled representations from grouped observations
    
    Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 32, no. 1 (2018)
    
10.  [Burgess et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb10)
    
    Burgess C.P., Higgins I., Pal A., Matthey L., Watters N., Desjardins G., Lerchner A.
    
    Understanding disentangling in β\-VAE
    
    (2018)
    
11.  [Campello et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb11)
    
    Campello V.M., Gkontra P., Izquierdo C., Martín-Isla C., Sojoudi A., Full P.M., Maier-Hein K., Zhang Y., He Z., Ma J., Parreño M., Albiol A., Kong F., Shadden S.C., Acero J.C., Sundaresan V., Saber M., Elattar M., Li H., Menze B., Khader F., Haarburger C., Scannell C.M., Veta M., Carscadden A., Punithakumar K., Liu X., Tsaftaris S.A., Huang X., Yang X., Li L., Zhuang X., Viladés D., Descalzo M.L., Guala A., Mura L.L., Friedrich M.G., Garg R., Lebel J., Henriques F., Karakas M., Çavuş E., Petersen S.E., Escalera S., Seguí S., Rodríguez-Palomares J.F., Lekadir K.
    
    Multi-centre, multi-vendor and multi-disease cardiac segmentation: The m&ms challenge
    
12.  [Carter and Nielsen, 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb12)
    
    Carter S., Nielsen M.
    
    Using artificial intelligence to augment human intelligence
    
13.  [Cetin et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb13)
    
    Cetin I., Raisi-Estabragh Z., Petersen S.E., Napel S., Piechnik S.K.P., Neubauer S., Gonzalez Ballester M.A., Camara O., Lekadir K.
    
    Radiomics signatures of cardiovascular risk factors in cardiac MRI: Results from the UK biobank
    
14.  [Cetin et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb14)
    
    Cetin I., Sanroma G., Petersen S.E., Napel S., Camara O., González Ballester M.A., Lekadir K.
    
    A radiomics approach to computer-aided diagnosis with cardiac cine-MRI
    
    Pop M., Sermesant M., Jodoin P., Lalande A., Zhuang X., Yang G., Young A.A., Bernard O. (Eds.), Statistical Atlases and Computational Models of the Heart. ACDC and MMWHS Challenges - 8th International Workshop, Lecture Notes in Computer Science, vol.10663, STACOM 2017, Held in Conjunction with MICCAI 2017, Quebec City, Canada, September 10-14, 2017, Revised Selected Papers, Springer (2017), pp. 82-90, [10.1007/978-3-319-75541-0\_9](https://doi.org/10.1007/978-3-319-75541-0_9)
    
15.  [Chartsias et al., 2019](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb15)
    
    Chartsias A., Joyce T., Papanastasiou G., Semple S., Williams M., Newby D.E., Dharmakumar R., Tsaftaris S.A.
    
    Disentangled representation learning in cardiac image analysis
    
    Med. Image Anal., 58 (2019), Article 101535
    
16.  [Chen et al., 2018a](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb16)
    
    Chen R.T.Q., Li X., Grosse R.B., Duvenaud D.K.
    
    Isolating sources of disentanglement in variational autoencoders
    
    Bengio S., Wallach H., Larochelle H., Grauman K., Cesa-Bianchi N., Garnett R. (Eds.), Advances in Neural Information Processing Systems, Vol. 31, Curran Associates, Inc. (2018)
    
17.  [Chen et al., 2018b](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb17)
    
    Chen W., Liu B., Peng S., Sun J., Qiao X.
    
    Computer-aided grading of Gliomas combining automatic segmentation and radiomics
    
18.  [Clough et al., 2019](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb18)
    
    Clough J.R., Oksuz I., Puyol-Antón E., Ruijsink B., King A.P., Schnabel J.A.
    
    Global and local interpretability for cardiac MRI classification
    
    Shen D., Liu T., Peters T.M., Staib L.H., Essert C., Zhou S., Yap P.-T., Khan A. (Eds.), Medical Image Computing and Computer Assisted Intervention, MICCAI 2019, Springer International Publishing, Cham (2019), pp. 656-664, [10.1007/978-3-030-32251-9\_72](https://doi.org/10.1007/978-3-030-32251-9_72)
    
19.  [Deng and Liu, 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb19)
    
    Deng L., Liu Y.
    
    Deep Learning in Natural Language Processing
    
    Springer Publishing Company, Incorporated (2018)
    
20.  [Ding et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb20)
    
    Ding, Z., Xu, Y., Xu, W., Parmar, G., Yang, Y., Welling, M., Tu, Z., 2020. Guided variational autoencoder for disentanglement learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7920–7929.
    
21.  [Girum et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb21)
    
    Girum K.B., Skandarani Y., Hussain R., Grayeli A.B., Créhange G., Lalande A.
    
    Automatic myocardial infarction evaluation from delayed-enhancement cardiac MRI using deep convolutional networks
    
    Puyol Anton E., Pop M., Sermesant M., Campello V., Lalande A., Lekadir K., Suinesiaputra A., Camara O., Young A. (Eds.), Statistical Atlases and Computational Models of the Heart. M&Ms and EMIDEC Challenges, Springer International Publishing, Cham (2021), pp. 378-384
    
22.  [Glorot and Bengio, 2010](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb22)
    
    Glorot X., Bengio Y.
    
    Understanding the difficulty of training deep feedforward neural networks
    
    Teh Y.W., Titterington M. (Eds.), Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, Proceedings of Machine Learning Research, vol.9, PMLR, Chia Laguna Resort, Sardinia, Italy (2010), pp. 249-256
    
23.  [Goodfellow et al., 2014](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb23)
    
    Goodfellow I., Pouget-Abadie J., Mirza M., Xu B., Warde-Farley D., Ozair S., Courville A., Bengio Y.
    
    Generative adversarial nets
    
    Ghahramani Z., Welling M., Cortes C., Lawrence N., Weinberger K.Q. (Eds.), Advances in Neural Information Processing Systems, Vol. 27, Curran Associates, Inc. (2014), pp. 2672-2680
    
24.  [Gretton et al., 2007](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb24)
    
    Gretton A., Borgwardt K., Rasch M., Schölkopf B., Smola A.
    
    A kernel method for the two-sample-problem
    
    Schölkopf B., Platt J., Hoffman T. (Eds.), Advances in Neural Information Processing Systems, Vol. 19, MIT Press (2007)
    
25.  [Hadjeres et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb25)
    
    Hadjeres G., Nielsen F., Pachet F.
    
    GLSR-VAE: Geodesic latent space regularization for variational autoencoder architectures
    
    2017 IEEE Symposium Series on Computational Intelligence, SSCI (2017), pp. 1-7, [10.1109/SSCI.2017.8280895](https://doi.org/10.1109/SSCI.2017.8280895)
    
26.  [Higaki et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb26)
    
    Higaki T., Nakamura Y., Zhou J., Yu Z., Nemoto T., Tatsugami F., Awai K.
    
    Deep learning reconstruction at CT : Phantom study of the image characteristics
    
27.  [Higgins et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb27)
    
    Higgins I., Matthey L., Pal A., Burgess C.P., Glorot X., Botvinick M.M., Mohamed S., Lerchner A.
    
    Beta-VAE: Learning basic visual concepts with a constrained variational framework
    
    ICLR (2017)
    
28.  [Huang et al., 2014](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb28)
    
    Huang M.L., Hung Y.H., Lee W.M., Li R.K., Jiang B.R.
    
    SVM-RFE based feature selection and Taguchi parameters optimization for multiclass SVM classifier
    
29.  [Ivantsits et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb29)
    
    Ivantsits M., Huellebrand M., Kelle S., Schönberg S.O., Kuehne T., Hennemuth A.
    
    Deep-learning-based myocardial pathology detection
    
    Puyol Anton E., Pop M., Sermesant M., Campello V., Lalande A., Lekadir K., Suinesiaputra A., Camara O., Young A. (Eds.), Statistical Atlases and Computational Models of the Heart. M&Ms and EMIDEC Challenges, Springer International Publishing, Cham (2021), pp. 369-377
    
30.  [Jo et al., 2019](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb30)
    
    Jo T., Nho K., Saykin A.J.
    
    Deep learning in Alzheimer’s disease: Diagnostic classification and prognostic prediction using neuroimaging data
    
31.  [Kapishnikov et al., 2019](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb31)
    
    Kapishnikov A., Bolukbasi T., Viegas F., Terry M.
    
    XRAI: Better Attributions Through Regions
    
    2019 IEEE/CVF International Conference on Computer Vision, ICCV (2019), pp. 4947-4956, [10.1109/ICCV.2019.00505](https://doi.org/10.1109/ICCV.2019.00505)
    
32.  [Kim and Mnih, 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb32)
    
    Kim H., Mnih A.
    
    Disentangling by factorising
    
    Dy J., Krause A. (Eds.), Proceedings of the 35th International Conference on Machine Learning, Proceedings of Machine Learning Research, vol.80, PMLR (2018), pp. 2649-2658
    
33.  [Kingma and Welling, 2014](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb33)
    
    Kingma D.P., Welling M.
    
    Auto-encoding variational Bayes
    
    2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings (2014)
    
34.  [Kofler et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb34)
    
    Kofler A., Haltmeier M., Schaeffter T., Kachelriess M., Dewey M., Wald C., Kolbitsch C.
    
    Neural networks-based regularization for large-scale medical image reconstruction
    
35.  [Kumar et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb35)
    
    Kumar A., Sattigeri P., Balakrishnan A.
    
    Variational inference of disentangled latent concepts from unlabeled observations
    
    (2017)
    
36.  [Lalande et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb36)
    
    Lalande A., Chen Z., Decourselle T., Qayyum A., Pommier T., Lorgis L., de la Rosa E., Cochet A., Cottin Y., Ginhac D., Salomon M., Couturier R., Meriaudeau F.
    
    Emidec: A database usable for the automatic evaluation of myocardial infarction from delayed-enhancement cardiac MRI
    
37.  [Lalande et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb37)
    
    Lalande A., Chen Z., Pommier T., Decourselle T., Qayyum A., Salomon M., Ginhac D., Skandarani Y., Boucher A., Brahim K., de Bruijne M., Camarasa R., Correia T., Feng X., Girum K.B., Hennemuth A., Huellebrand M., Hussain R., Ivantsits M., Ma J., Meyer C.H., Sharma R., Shi J., Tsekos N.V., Varela M., Wang X., Yang S., Zhang H., Zhang Y., Zhou Y., Zhuang X., Couturier R., Mériaudeau F.
    
    Deep learning methods for automatic evaluation of delayed enhancement-MRI. the results of the EMIDEC challenge
    
    (2021)
    
38.  [Lambin et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb38)
    
    Lambin P., Leijenaar R.T.H., Deist T.M., Peerlings J., de Jong E.E.C., van Timmeren J., Sanduleanu S., Larue R.T.H.M., Even A.J.G., Jochems A., van Wijk Y., Woodruff H., van Soest J., Lustberg T., Roelofs E., van Elmpt W., Dekker A., Mottaghy F.M., Wildberger J.E., Walsh S.
    
    Radiomics: The bridge between medical imaging and personalized medicine
    
39.  [Lample et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb39)
    
    Lample G., Zeghidour N., Usunier N., Bordes A., Denoyer L., Ranzato M.
    
    Fader networks: Manipulating images by sliding attributes
    
    Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS ’17, Curran Associates Inc., Red Hook, NY, USA (2017), pp. 5969-5978
    
40.  [Larroza et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb40)
    
    Larroza A., López-Lereu M.P., Monmeneu J.V., Gavara J., Chorro F.J., Bodí V., Moratal D.
    
    Texture analysis of cardiac cine magnetic resonance imaging to detect nonviable segments in patients with chronic myocardial infarction
    
41.  [Liu et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb41)
    
    Liu W., Li R., Zheng M., Karanam S., Wu Z., Bhanu B., Radke R.J., Camps O.I.
    
    Towards visually explaining variational autoencoders
    
    2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR (2020), pp. 8639-8648, [10.1109/CVPR42600.2020.00867](https://doi.org/10.1109/CVPR42600.2020.00867)
    
42.  [Locatello et al., 2019](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb42)
    
    Locatello F., Bauer S., Lucic M., Raetsch G., Gelly S., Schölkopf B., Bachem O.
    
    Challenging common assumptions in the unsupervised learning of disentangled representations
    
    Chaudhuri K., Salakhutdinov R. (Eds.), Proceedings of the 36th International Conference on Machine Learning, Proceedings of Machine Learning Research, vol.97, PMLR (2019), pp. 4114-4124
    
43.  [López-Linares et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb43)
    
    López-Linares K., Aranjuelo N., Kabongo L., Maclair G., Lete N., Ceresa M., García-Familiar A., Macía I., González Ballester M.A.
    
    Fully automatic detection and segmentation of abdominal aortic thrombus in post-operative CTA images using deep convolutional neural networks
    
44.  [Lourenço et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb44)
    
    Lourenço A., Kerfoot E., Grigorescu I., Scannell C.M., Varela M., Correia T.M.
    
    Automatic Myocardial disease prediction from delayed-enhancement cardiac MRI and clinical information
    
    Puyol Anton E., Pop M., Sermesant M., Campello V., Lalande A., Lekadir K., Suinesiaputra A., Camara O., Young A. (Eds.), Statistical Atlases and Computational Models of the Heart. M&Ms and EMIDEC Challenges, Springer International Publishing, Cham (2021), pp. 334-341
    
45.  [Lundberg and Lee, 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb45)
    
    Lundberg S.M., Lee S.-I.
    
    A unified approach to interpreting model predictions
    
    Guyon I., Luxburg U.V., Bengio S., Wallach H., Fergus R., Vishwanathan S., Garnett R. (Eds.), Advances in Neural Information Processing Systems, Vol. 30, NIPS ’17, Curran Associates, Inc. (2017), pp. 4768-4777
    
46.  [Masis, 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb46)
    
    Masis S.
    
    Interpretable Machine Learning with Python
    
    Packt Publishing (2021)
    
47.  [McCrindle et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb47)
    
    McCrindle B., Zukotynski K., Doyle T.E., Noseworthy M.D.
    
    A radiology-focused review of predictive uncertainty for AI interpretability in computer-assisted segmentation
    
48.  [Molnar, 2022](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb48)
    
    Interpretable Machine Learning (second ed.) (2022)
    
49.  [Neisius et al., 2019](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb49)
    
    Neisius U., El-Rewaidy H., Nakamori S., Rodriguez J., Manning W.J., Nezafat R.
    
    Radiomic analysis of myocardial native T1 imaging discriminates between hypertensive heart disease and hypertrophic cardiomyopathy
    
50.  [Painchaud et al., 2022](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb50)
    
    Painchaud N., Duchateau N., Bernard O., Jodoin P.-M.
    
    Echocardiography segmentation with enforced temporal consistency
    
51.  [Pati and Lerch, 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb51)
    
    Pati A., Lerch A.
    
    Attribute-based regularization of latent spaces for variational auto-encoders
    
52.  [Pitale et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb52)
    
    Pitale R., Kale H., Kshirsagar S., Rajput H.
    
    A schematic review on applications of deep learning and computer vision
    
53.  [Puyol-Antón et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb53)
    
    Puyol-Antón E., Chen C., Clough J.R., Ruijsink B., Sidhu B.S., Gould J., Porter B., Elliott M., Mehta V., Rueckert D., Rinaldi C.A., King A.P.
    
    Interpretable deep models for cardiac resynchronisation therapy response prediction
    
    Martel A.L., Abolmaesumi P., Stoyanov D., Mateus D., Zuluaga M.A., Zhou S.K., Racoceanu D., Joskowicz L. (Eds.), Medical Image Computing and Computer Assisted Intervention, MICCAI 2020, Springer International Publishing, Cham (2020), pp. 284-293, [10.1007/978-3-030-59710-8\_28](https://doi.org/10.1007/978-3-030-59710-8_28)
    
54.  [Reyes et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb54)
    
    Reyes M., Meier R., Pereira S., Silva C.A., Dahlweid F.M., von Tengg-Kobligk H., Summers R.M., Wiest R.
    
    On the interpretability of artificial intelligence in radiology: Challenges and opportunities
    
55.  [Ribeiro et al., 2016](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb55)
    
    Ribeiro M.T., Singh S., Guestrin C.
    
    ”Why should I trust you?”: Explaining the predictions of any classifier
    
    Proceedings of the Demonstrations Session, NAACL HLT 2016, the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, San Diego California, USA, June 12-17, 2016, The Association for Computational Linguistics (2016), pp. 97-101, [10.18653/v1/n16-3020](https://doi.org/10.18653/v1/n16-3020)
    
56.  [Ridgeway and Mozer, 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb56)
    
    Ridgeway K., Mozer M.C.
    
    Learning deep disentangled embeddings with the F-statistic loss
    
    Bengio S., Wallach H., Larochelle H., Grauman K., Cesa-Bianchi N., Garnett R. (Eds.), Advances in Neural Information Processing Systems, Vol. 31, Curran Associates, Inc. (2018), pp. 185-194
    
57.  [Ronneberger et al., 2015](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb57)
    
    Ronneberger O., Fischer P., Brox T.
    
    U-Net: Convolutional networks for biomedical image segmentation
    
    Navab N., Hornegger J., Wells W.M., Frangi A.F. (Eds.), Medical Image Computing and Computer-Assisted Intervention, MICCAI 2015, Springer International Publishing, Cham (2015), pp. 234-241, [10.1007/978-3-319-24574-4\_28](https://doi.org/10.1007/978-3-319-24574-4_28)
    
58.  [Rubenstein et al., 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb58)
    
    Rubenstein P.K., Schölkopf B., Tolstikhin I.O.
    
    Learning disentangled representations with wasserstein auto-encoders
    
    ICLR (2018)
    
59.  [Rudin, 2019](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb59)
    
    Rudin C.
    
    Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead
    
60.  [Salahuddin et al., 2022](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb60)
    
    Salahuddin Z., Woodruff H.C., Chatterjee A., Lambin P.
    
    Transparency of deep neural networks for medical image analysis: A review of interpretability methods
    
61.  [Samb et al., 2012](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb61)
    
    Samb M.L., Camara F., Ndiaye S., Slimani Y., Esseghir M.A., Anta C.
    
    A novel RFE-SVM-based feature selection approach for classification
    
62.  [Sanchez-Martinez et al., 2022](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb62)
    
    Sanchez-Martinez S., Camara O., Piella G., Cikes M., González Ballester M.A., Miron V.A., Gomez E., Fraser A., Bijnens B.
    
    Machine learning for clinical decision-making: Challenges and opportunities in cardiovascular imaging
    
63.  [Selvaraju et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb63)
    
    Selvaraju R.R., Cogswell M., Das A., Vedantam R., Parikh D., Batra D.
    
    Grad-CAM: Visual explanations from deep networks via gradient-based localization
    
    2017 IEEE International Conference on Computer Vision, ICCV (2017), pp. 618-626, [10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74)
    
64.  [Shakeri et al., 2016](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb64)
    
    Shakeri M., Lombaert H., Tripathi S., Kadoury S.
    
    Deep spectral-based shape features for alzheimer’s disease classification
    
    Reuter M., Wachinger C., Lombaert H. (Eds.), Spectral and Shape Analysis in Medical Imaging, Springer International Publishing, Cham (2016), pp. 15-24, [10.1007/978-3-319-51237-2\_2](https://doi.org/10.1007/978-3-319-51237-2_2)
    
65.  [Sharma et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb65)
    
    Sharma R., Eick C.F., Tsekos N.V.
    
    SM2n2: A stacked architecture for multimodal data and its application to myocardial infarction detection
    
    Puyol Anton E., Pop M., Sermesant M., Campello V., Lalande A., Lekadir K., Suinesiaputra A., Camara O., Young A. (Eds.), Statistical Atlases and Computational Models of the Heart. M&Ms and EMIDEC Challenges, Springer International Publishing, Cham (2021), pp. 342-350
    
66.  [Shen et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb66)
    
    Shen D., Wu G., Suk H.I.
    
    Deep learning in medical image analysis
    
67.  [Shi et al., 2021](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb67)
    
    Shi J., Chen Z., Couturier R.
    
    Classification of pathological cases of myocardial infarction using convolutional neural network and random forest
    
    Puyol Anton E., Pop M., Sermesant M., Campello V., Lalande A., Lekadir K., Suinesiaputra A., Camara O., Young A. (Eds.), Statistical Atlases and Computational Models of the Heart. M&Ms and EMIDEC Challenges, Springer International Publishing, Cham (2021), pp. 406-413
    
68.  [Simonyan et al., 2014](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb68)
    
    Simonyan K., Vedaldi A., Zisserman A.
    
    Deep inside convolutional networks: Visualising image classification models and Saliency maps
    
    Bengio Y., LeCun Y. (Eds.), 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Workshop Track Proceedings (2014)
    
69.  [Singh et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb69)
    
    Singh A., Sengupta S., Lakshminarayanan V.
    
    Explainable deep learning models in medical image analysis
    
70.  [Wu et al., 2019](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb70)
    
    Wu S., Roberts K., Datta S., Du J., Ji Z., Si Y., Soni S., Wang Q., Wei Q., Xiang Y., Zhao B., Xu H.
    
    Deep learning in clinical natural language processing: a methodical review
    
71.  [Xiao et al., 2020](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb71)
    
    Xiao G., Rong W.C., Hu Y.C., Shi Z.Q., Yang Y., Ren J.L., Cui G.B.
    
    MRI radiomics analysis for predicting the pathologic classification and TNM staging of thymic epithelial Tumors: A pilot study
    
72.  [Yang, 2018](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb72)
    
    Yang X.
    
    Identification of risk genes associated with myocardial infarction based on the recursive feature elimination algorithm and support vector machine classifier
    
73.  [Zhang et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb73)
    
    Zhang X., Xu X., Tian Q., Li B., Wu Y., Yang Z., Liang Z., Liu Y., Cui G., Lu H.
    
    Radiomics assessment of bladder cancer grade using texture features from diffusion-weighted imaging
    
    J. Magn. Reson. Imaging, 46 (5) (2017), pp. 1281-1288, [10.1002/jmri.25669](https://doi.org/10.1002/jmri.25669)
    
74.  [Zhu et al., 2017](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bb74)
    
    Zhu J.-Y., Park T., Isola P., Efros A.A.
    
    Unpaired image-to-image translation using cycle-consistent adversarial networks
    
    2017 IEEE International Conference on Computer Vision, ICCV (2017), pp. 2242-2251, [10.1109/ICCV.2017.244](https://doi.org/10.1109/ICCV.2017.244)
    

## Cited by (32)

[<sup>1</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bfn1)

[<sup>2</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bfn2)

[<sup>3</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bfn3)

A score based on physical examination and the development of the heart failure to predict the risk of mortality.

[<sup>4</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bfn4)

A parameter that shows the level of the protein that is released into the bloodstream.

[<sup>5</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bfn5)

A parameter that shows a level of a peptide, which is an indicator for the diagnosis of heart failure.

[<sup>6</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bfn6)

[<sup>7</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bfn7)

[<sup>8</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bfn8)

[<sup>9</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bfn9)

[<sup>10</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bfn10)

[<sup>11</sup>](https://www.sciencedirect.com/science/article/pii/S0895611122001288#bfn11)

© 2023 The Authors. Published by Elsevier Ltd.