# CSCI599 Final Course Project

## Papers

- Must-read
  * [Playing against Nature: causal discovery for decision making under uncertainty](https://arxiv.org/pdf/1807.01268.pdf)
    - **Extremely important!!** Seems very similar to what we are doing
  * [Causal Reasoning from Meta-reinforcement Learning](https://arxiv.org/pdf/1901.08162.pdf)
    - *why interesting*: The work that inspires our ideas
  * [Woulda, Coulda, Shoulda: Counterfactually-Guided Policy Search](https://openreview.net/forum?id=BJG0voC9YQ)
    - Introduce Counterfactually-Guided Policy Search (CF-GPS) algorithm to leverage Structural Causal Models (SCM) for counterfactual evaluation of alternative policies.
    - *why interesting*: Using SCM to guide policy learning may be regarded as an extension of our work. Need to know in this paper whether the SCM is given as a prior knowledge of the environment or learned from the environment.
  * [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/pdf/1812.08434.pdf)
- Worth reading
  * [Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/pdf/1805.00909.pdf)
    - A framework that generalizes optimal control or reinforcement learning problems to the exact inference problem in probabilistic graphical models (PGMs). In this way, learning the optimal policy in RL is equivalent to inference on the special probabilistic graphical models. 
    - *why interesting*: This is a potentially very influential paper since it unifies RL with PGMs. This may guide or even alter the design of our system in a fundamental way.
  * [VIME: Variational Information Maximizing Exploration](https://arxiv.org/abs/1605.09674)
    - This paper introduces Variational Information Maximizing Exploration (VIME), an exploration strategy based on maximization of information gain about the agent's belief of environment dynamics. We propose a practical implementation, using variational inference in Bayesian neural networks which efficiently handles continuous state and action spaces.
  * [Causal Generative Neural Networks](https://arxiv.org/pdf/1711.08936.pdf)
  * [Reinforcement learning and causal models](http://gershmanlab.webfactional.com/pubs/RL_causal.pdf)
    - The first half of the chapter contrasts a “model-free” system that learns to repeat actions that lead to reward with a “model-based” system that learns a probabilistic causal model of the environment which it then uses to plan action sequences.
    - *why interesting*: An 2015 paper that may provide some high-level inspirations, particularly how a "model-based" RL system learns a probabilistic causal model.
  * [Causal Learning versus Reinforcement Learning for Knowledge Learning and Problem Solving](https://aaai.org/ocs/index.php/WS/AAAIW17/paper/view/15182/14741)
    - (abstract) Causal learning and reinforcement learning are both important AI learning mechanisms but are usually treated separately, despite the fact that both are directly relevant to problem solving processes. In this paper we propose a method for causal learning and problem solving, and compare and contrast that with AI reinforcement learning and show that the two methods are actually related, differing only in the values of the learning rate α and discount factor γ. However, the causal learning framework emphasizes quick but non-optimal concoction of problem solutions while AI reinforcement learning generates optimal solutions at the expense of speed. Cognitive science literature is reviewed and it is found that psychological reinforcement learning in lower form animals such as mammals is distinct from AI reinforcement learning in that psychological reinforcement learning strives neither for speed nor optimality, and that higher form animals such as humans and primates employ quick causal learning for survival instead of reinforcement learning. AI systems should likewise take advantage of a framework that employs rapid inductive causal learning to generate problem solutions for its general viability in terms of rapid adaptability, without the need to always strive for optimality.
  * [Learning Plannable Representations with Causal InfoGAN](https://arxiv.org/pdf/1807.09341.pdf), [medium summary](https://medium.com/arxiv-bytes/summary-learning-plannable-representations-with-causal-infogan-c357433b19be)
    - The focus of this work is to go about planning a sequence of abstract states towards a goal and then decode the abstract states to their corresponding predicted observations. This is contrary to many other popular methods that perform planning over a sequence of actions. The approach to planning was in part inspired by [InfoGAN](https://arxiv.org/pdf/1606.03657.pdf).
  * [Discovering latent causes in reinforcement learning](https://www.princeton.edu/~nivlab/papers/GershmanNormanNiv2015.pdf)
    - A behavioral science paper. May provide high-level intuitions but don't rely on it for implementation details
  * [Learning model-based planning from scratch](https://arxiv.org/pdf/1707.06170.pdf)
    - Imagination-based planning. Model-based reinforcement learning.
    - *why interesting*: see how similar it is to learning a SCM
  * [Combined Reinforcement Learning via Abstract Representations](https://arxiv.org/pdf/1809.04506.pdf)
    - In the quest for efficient and robust reinforcement learning methods, both model-free and model-based approaches offer advantages. In this paper we propose a new way of explicitly bridging both approaches via a shared low-dimensional learned encoding of the environment, meant to capture summarizing abstractions.
    - *why interesting*: May shed lights on learning representations from the environment.
  * [Characterizing and Learning Equivalence Classes of Causal DAGs under Interventions](https://arxiv.org/pdf/1802.06310.pdf)
    - We consider the problem of learning causal DAGs in the setting where both observational and interventional data is available. This setting is common in biology, where gene regulatory networks can be intervened on using chemical reagents or gene deletions. Hauser & Buhlmann (2012) previously characterized the identifiability of causal DAGs under perfect interventions, which eliminate dependencies between targeted variables and their direct causes. In this paper, we extend these identifiability results to general interventions, which may modify the dependencies between targeted variables and their causes without eliminating them. We define and characterize the interventional Markov equivalence class that can be identified from general (not necessarily perfect) intervention experiments. We also propose the first provably consistent algorithm for learning DAGs in this setting and evaluate our algorithm on simulated and biological datasets.
  * [Causal Confusion in Imitation Learning](https://people.eecs.berkeley.edu/~dineshjayaraman/projects/causal_confusion_nips18.pdf)
    - Shows that causally-unaware imitation learning is bad. We propose a solution to combat causal confusion, which involves first inferring a distribution over potential causal models consistent with the behavioral cloning objective, then identifying the correct hypothesis through “intervention”. Our approach permits intervention in the form either of expert queries or of policy execution in the environment.
    - *why interesting*: Although in the field of imitation learning not reinforcement learning, this work actually propose a way to learn causal models either of the "expert" or of the environment itself.
  * [Modularization of End-to-End Learning: Case Study in Arcade Games](https://arxiv.org/pdf/1901.09895.pdf)
    - Complex environments and tasks pose a difficult problem for holistic end-to-end learning approaches. Decomposition of an environment into interacting controllable and non-controllable objects allows supervised learning for non-controllable objects and universal value function approximator learning for controllable objects. Such decomposition should lead to a shorter learning time and better generalization capability. Here, we consider arcade-game environments as sets of interacting objects (controllable, non-controllable) and propose a set of functional modules that are specialized on mastering different types of interactions in a broad range of environments.
  * [Rule Discovery for Exploratory Causal Reasoning](http://eda.mmci.uni-saarland.de/pubs/2018/dice-budhathoki,boley.vreeken-nipscl.pdf)
    - Not really as interesting because it focuses on learning causal rules from observational data, whereas our projects aims to learn that from an interactive environment which permits interventions. However, learning causal rules in general may still be helpful.
  * [Representation Balancing MDPs for Off-Policy Policy Evaluation](https://arxiv.org/pdf/1805.09044.pdf)
    - Not really as interesting
- Background & Introduction
  * [Machine Learning Tutorial Series @ Imperial College](http://www.homepages.ucl.ac.uk/~ucgtrbd/talks/imperial_causality.pdf)
    - General introduction to causal inference
