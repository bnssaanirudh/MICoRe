# MICoRe: Identifiable Causal Representation Learning via Sparse Soft Interventions and Minimal Intervention Regularization

**Abstract**
Learning identifiable causal representations from high-dimensional observations is a fundamental challenge in AI. We propose MICoRe, a framework that combines Identifiable Variational Autoencoders (iVAE) with continuous DAG learning (NOTEARS) and a novel Minimal Intervention Regularization loss. Our method leverages the sparsity of interventions across different environments to enforce the recovery of the true causal factors and their structural relationships. We evaluate MICoRe on synthetic and real-world datasets, demonstrating superior disentanglement and causal graph accuracy compared to standard VAE and IRM baselines.

## 1. Introduction
Traditional representation learning often results in entangled latent spaces. Causal Representation Learning (CRL) aims to recover the underlying causal variables. Current methods either require hard interventions or assume a known graph. MICoRe addresses these limitations by learning both the latents and the graph under sparse soft interventions across different environments.

## 2. Methodology

### 2.1 The "u" Variable Binding & Latent Representation Module
In our framework, the auxiliary variable $u$ is explicitly bound to the environment index. We use an iVAE-based architecture where the encoder $q(z|x, u)$ maps observations to a latent space conditioned on $u$.

### 2.2 Resolving the Architecture Collision
A standard iVAE assumes the latents $Z$ are marginally independent, which contradicts the goal of learning a causal graph $W$ using NOTEARS. To unify these frameworks, we learn a Structural Causal Model (SCM) over the latents: $Z = f(Z; W) + \epsilon$. 
Crucially, the iVAE's conditionally independent prior is placed on the **exogenous noise variables** $\epsilon$, not $Z$. Because $W$ forms a directed acyclic graph (DAG), the Jacobian determinant of the transformation $\epsilon \to Z$ is 1, and the likelihood becomes $\log p(Z|u, W) = \sum_i \log p(\epsilon_i | u)$. The causal structure $W$ is regularized by the continuous DAG penalty: $tr(e^{W \circ W}) - d = 0$.

### 2.3 Minimal Intervention Regularization
To exploit the multi-environment nature of the data, we introduce a sparsity penalty applied to the shift in the causal mechanisms. Let $\theta_{i,u}$ be the prior parameters (mean and log-variance) for the exogenous noise of variable $i$ in environment $u$. We define the Minimal Intervention Loss as:
$$L_{MI} = \lambda \sum_{u > 0} \sum_i ||\theta_{i,u} - \theta_{i,0}||_1$$
This penalty forces the mechanisms of most variables to remain invariant (shift = 0) between the observational environment ($u=0$) and interventional environments ($u > 0$), perfectly capturing the sparse soft intervention assumption.

## 3. Experiments

### 3.1 Datasets
- **Causal3DIdent**: Synthetic 3D scenes with causal factors.
- **Pendulum**: Coupled physical systems.
- **Sachs**: Protein signaling network.

### 3.2 Evaluation Metrics
We rely on canonical metrics for identifiability and graph recovery:
- **Mean Correlation Coefficient (MCC)**: To prove the iVAE successfully recovered the true latents up to permutation and scaling.
- **Structural Hamming Distance (SHD)**: To evaluate how accurately NOTEARS recovered the directed edges.

## 4. Results & Discussion
Our experiments show that MICoRe successfully recovers the causal graph in the Pendulum dataset with low SHD and achieves high MCC scores in synthetic tasks, proving the efficacy of evaluating independent priors over the exogenous residuals rather than the latents themselves.

## 5. Conclusion
MICoRe provides a robust and theoretically sound framework for CRL, proving that minimal intervention assumptions and exogenous priors are powerful tools for identifiability.
